use std::{io, collections::HashMap};
use std::{
    ffi::CString,
    fs::File,
    path::Path,
};

use rspirv::{
    binary::Parser,
    dr::{Block, Function, Instruction, Loader, Module, Operand},
};
use serde::{Deserialize, Serialize};
pub use spirv::ExecutionModel;
use spirv::Decoration;

#[cfg(feature = "codegen")]
pub mod codegen;
#[cfg(feature = "convert")]
pub mod convert;
pub mod data;
pub(crate) mod ext;
pub mod types;
pub mod values;
pub(crate) mod util;

use data::*;

use crate::types::{SpirvType, SpirvTypeOwned};
use ext::ModuleExt;

macro_rules! expect_operand {
    (&$source: expr, $index: literal, $operand: path) => {
        match ($source).get($index) {
            Some($operand(it)) => *it,
            Some(other) => {
                return Err(Error::UnexpectedOperand {
                    index: $index,
                    expected_type: stringify!($operand),
                    found: other.clone(),
                })
            }
            None => {
                return Err(Error::MissingOperand {
                    index: $index,
                    expected_type: stringify!($operand),
                })
            }
        }
    };
    ($source: expr, $index: literal, $operand: path) => {
        expect_operand(&source, $index, $operand)
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingDescription {
    pub set: u32,
    pub binding: u32,
    pub count: Option<u32>,
    pub descriptor_type: DescriptorType,
    pub name: Option<String>,
    #[serde(skip)]
    pub spirv_type_ref: u32,
}

impl BindingDescription {
    pub fn all_from_module(module: &ModuleExt) -> Result<Vec<BindingDescription>, Error> {
        let mut result = Vec::new();
        for descriptor in module.decorated_instructions(Decoration::Binding) {
            let reference = expect_operand!(&descriptor.operands, 0, Operand::IdRef);
            let binding = expect_operand!(&descriptor.operands, 2, Operand::LiteralBit32);
            let set = match module.ref_decoration(reference, Decoration::DescriptorSet) {
                Some(&[Operand::LiteralBit32(value)]) => value,
                _ => panic!("binding missing set decoration"),
            };
            let name = module.ref_name(reference).map(str::to_string);
            let storage_class = module
                .ref_storage_class(reference)
                .ok_or_else(|| Error::UnknownStorageClass(reference))?;
            let info = module
                .descriptor_info_for(reference, storage_class)
                .ok_or_else(|| Error::MissingStorageInfo(reference))?;
            result.push(BindingDescription {
                set,
                binding,
                count: info.binding_count.map(|it| it as u32),
                descriptor_type: info.ty,
                name,
                spirv_type_ref: reference,
            })
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushConstantDescription {
    pub name: Option<String>,
    #[serde(skip)]
    pub spirv_type_ref: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EntryPoint {
    pub stage: ShaderStage,
    pub name: CString,
    pub bindings: Vec<BindingDescription>,
    pub push_constant: Option<PushConstantDescription>,
    #[serde(skip)]
    pub spirv_function_ref: u32,
}

impl TryFrom<&Instruction> for EntryPoint {
    type Error = std::io::Error;
    fn try_from(value: &Instruction) -> Result<Self, Self::Error> {
        let model = value
            .operands
            .first()
            .and_then(|it| match it {
                rspirv::dr::Operand::ExecutionModel(it) => Some(it),
                _ => None,
            })
            .cloned()
            .ok_or_else(|| {
                std::io::Error::new(io::ErrorKind::InvalidData, Error::MissingEntryPointModel)
            })?;

        let function_ref = value
            .operands
            .get(1)
            .and_then(|it| match it {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .ok_or_else(|| {
                std::io::Error::new(
                    io::ErrorKind::InvalidData,
                    Error::MissingEntryPointFunctionRef,
                )
            })?;

        let name = value
            .operands
            .get(2)
            .and_then(|it| match it {
                rspirv::dr::Operand::LiteralString(it) => Some(CString::new(it.clone()).unwrap()),
                _ => None,
            })
            .ok_or_else(|| {
                std::io::Error::new(io::ErrorKind::InvalidData, Error::MissingEntryPointName)
            })?;

        Ok(Self {
            stage: ShaderStage::from(model),
            name,
            bindings: Vec::new(),
            push_constant: None,
            spirv_function_ref: function_ref,
        })
    }
}

struct EntryPointVisitor<'a, 'm> {
    entry_point: &'a mut EntryPoint,
    bindings: &'a [BindingDescription],
    module: &'m ModuleExt,
    visited: Vec<*const u8>,
}
impl<'a, 'm> EntryPointVisitor<'a, 'm> {
    fn new(
        entry_point: &'a mut EntryPoint,
        module: &'m ModuleExt,
        bindings: &'a [BindingDescription],
    ) -> Self {
        EntryPointVisitor {
            entry_point,
            bindings,
            module,
            visited: Vec::new(),
        }
    }

    fn attach_ref(&mut self, spirv_type_ref: u32) {
        let binding = self.bindings.iter().find(|it| it.spirv_type_ref == spirv_type_ref);
        if let Some(binding) = binding {
            self.entry_point.bindings.push(binding.clone());
            return;
        }

        if let Some(StorageClass::PushConstant) = self.module.ref_storage_class(spirv_type_ref) {
            if let Some(pc_ref) = &self.entry_point.push_constant {
                assert_eq!(
                    spirv_type_ref, pc_ref.spirv_type_ref,
                    "only one push constant definition allowed"
                );
                return;
            }
            self.entry_point.push_constant = Some(PushConstantDescription {
                name: self.module.ref_name(spirv_type_ref).map(str::to_string),
                spirv_type_ref,
            });
        }
    }
    fn mark_visited<T>(&mut self, obj: &'m T) {
        self.visited.push((obj as *const T).cast())
    }
    fn is_visited<T>(&mut self, obj: &'m T) -> bool {
        self.visited.contains(&(obj as *const T).cast())
    }

    fn visit_instruction(&mut self, instruction: &'m Instruction) {
        match instruction.class.opcode {
            spirv::Op::Branch | spirv::Op::BranchConditional | spirv::Op::Switch => {
                // We visit all blocks in visit_fn so no need to resolve references and traverse in order
            }
            spirv::Op::FunctionCall => {
                let callee: u32 = instruction
                    .operands
                    .first()
                    .map(|it| match it {
                        rspirv::dr::Operand::IdRef(it) => *it,
                        _ => unreachable!("unexpected callee reference operand"),
                    })
                    .expect("function missing callee reference");
                self.attach_ref(callee);
                if let Some(callee) = self
                    .module
                    .functions
                    .iter()
                    .find(|it| it.def_id() == Some(callee))
                {
                    self.visit_fn(callee)
                    // else: function not in current module
                }
            }
            _ => {
                let refs: Vec<u32> = instruction
                    .operands
                    .iter()
                    .filter_map(|it| match it {
                        rspirv::dr::Operand::IdRef(it) => Some(*it),
                        _ => None,
                    })
                    .collect();
                for r in refs {
                    self.attach_ref(r);
                }
                if let Some(result) = instruction.result_id {
                    self.attach_ref(result);
                }
            }
        }
    }

    fn visit_block(&mut self, block: &'m Block) {
        if self.is_visited(block) {
            return;
        }
        for instruction in &block.instructions {
            self.visit_instruction(instruction);
        }
        self.mark_visited(block);
    }

    fn visit_fn(&mut self, f: &'m Function) {
        if self.is_visited(f) {
            return;
        }
        for block in &f.blocks {
            self.visit_block(block);
        }
        self.mark_visited(f);
    }

    fn visit(&mut self) -> Result<(), Error> {
        let function: &Function = self
            .module
            .functions
            .iter()
            .find(|it| it.def_id() == Some(self.entry_point.spirv_function_ref))
            .ok_or_else(|| Error::MissingEntryPointFunction(self.entry_point.name.clone()))?;
        self.visit_fn(function);
        Ok(())
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub entry_points: Vec<EntryPoint>,
    pub types: HashMap<u32, SpirvTypeOwned>,
}
impl ModuleMetadata {
    pub fn new(spirv_bytecode: impl AsRef<[u32]>) -> Result<Self, Error> {
        let source = spirv_bytecode.as_ref();

        let module = {
            let mut loader = Loader::new();
            let code: &[u8] = unsafe {
                std::slice::from_raw_parts(source.as_ptr().cast(), source.len() * 4)
            };
            let p = Parser::new(code, &mut loader);
            p.parse().map_err(|_| Error::InvalidSpirvBinary)?;
            ModuleExt::new(loader.module())
        };

        let descriptors = BindingDescription::all_from_module(&module)?;

        let mut entry_points = Vec::new();
        for entry in &module.entry_points {
            let mut entry_point = EntryPoint::try_from(entry)?;
            let mut populator = EntryPointVisitor::new(&mut entry_point, &module, &descriptors);
            populator.visit()?;
            entry_points.push(entry_point);
        }

        Ok(ModuleMetadata {
            entry_points,
            types: module.types,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpirVModule {
    pub metadata: ModuleMetadata,
    pub source: Vec<u32>,
}

impl SpirVModule {
    pub fn open(file: impl AsRef<Path>) -> Result<Self, Error> {
        let mut source = File::open(file.as_ref())?;
        Self::read(&mut source)
    }

    pub fn read<R: io::Read + io::Seek>(reader: &mut R) -> Result<Self, Error> {
        let source: Vec<u32> = util::read_spv(reader)?;

        Ok(Self {
            metadata: ModuleMetadata::new(&source)?,
            source,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SPIR-V binary is invalid")]
    InvalidSpirvBinary,
    #[error("entry point is missing execution model argument")]
    MissingEntryPointModel,
    #[error("entry point is missing function reference argument")]
    MissingEntryPointFunctionRef,
    #[error("missing operand #{index} of type {expected_type}")]
    MissingOperand {
        index: usize,
        expected_type: &'static str,
    },
    #[error("expected operand #{index} to be of type {expected_type}; found {found:?}")]
    UnexpectedOperand {
        index: usize,
        expected_type: &'static str,
        found: Operand,
    },
    #[error("can't infer storage details for binding {0}")]
    MissingStorageInfo(u32),
    #[error("binding {0} has unknown storage class")]
    UnknownStorageClass(u32),
    #[error("unable to locate entry point function for {0:?}")]
    MissingEntryPointFunction(CString),
    #[error("entry point is missing name argument")]
    MissingEntryPointName,

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
