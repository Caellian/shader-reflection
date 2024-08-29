use std::{ops::Deref, collections::HashMap, cell::RefCell, hash::Hash};

use rspirv::dr::{Instruction, Module, ModuleHeader, Operand};
use spirv::{Decoration, Op, StorageClass};

use crate::{DescriptorType, SpirVModule, types::*, values::*, ImageSampling};

macro_rules! match_operand {
    ($source: expr, $index: literal, $operand: path) => {
        match ($source).get($index) {
            Some($operand(it)) => Some(*it),
            _ => None,
        }
    };
}

macro_rules! expect_operand {
    ($source: expr, $index: literal, $operand: ident, $name: literal) => {
        match_operand!($source, $index, Operand::$operand).expect(concat![
            "missing ",
            $name,
            " (",
            stringify!($operand),
            ") operand",
        ])
    };
}

pub struct ModuleExt {
    inner: Module,
    pub(crate) types: HashMap<u32, SpirvTypeOwned>,
    pub(crate) values: HashMap<u32, SpirvValue>,
}

impl Deref for ModuleExt {
    type Target = Module;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ModuleExt {
    pub(crate) fn new(module: Module) -> Self {
        let mut result = ModuleExt {
            inner: module,
            types: HashMap::new(),
            values: HashMap::new(),
        };

        for instruction in &result.inner.types_global_values {
            let id = match instruction.result_id {
                Some(it) => it,
                None => continue
            };

            if let Some(ty) = SpirvTypeOwned::from_instruction(&result, instruction, None) {
                result.types.insert(id, ty);
                continue;
            } 
            
            let result_type = instruction.result_type.and_then(|it| result.types.get(&it));
            if let (Op::Variable, Some(ty_res)) = (instruction.class.opcode, result_type) {
                result.types.insert(id, ty_res.clone());
            }
        }

        result
    }

    pub fn version(&self) -> (u8, u8) {
        self.header
            .as_ref()
            .map(ModuleHeader::version)
            .expect("missing SPIRV header version")
    }

    pub fn decorated_instructions(&self, which: Decoration) -> impl Iterator<Item = &Instruction> + '_ {
        self.annotations.iter().filter(move |it| {
            it.class.opcode == Op::Decorate
                && it
                    .operands
                    .get(1)
                    .map(|it| matches!(it, rspirv::dr::Operand::Decoration(decoration) if *decoration == which))
                    .unwrap_or_default()
        })
    }

    pub fn ref_decorations(&self, id: u32) -> impl Iterator<Item = (Decoration, &'_ [Operand])> + '_ {
        self.annotations
            .iter()
            .filter(move |it| {
                it.class.opcode == Op::Decorate
                    && it
                        .operands
                        .first()
                        .map(|it| *it == Operand::IdRef(id))
                        .unwrap_or_default()
            })
            .map(|it| {
                (
                    it.operands
                        .get(1)
                        .and_then(|it| match it {
                            Operand::Decoration(it) => Some(*it),
                            _ => None,
                        })
                        .unwrap(),
                    &it.operands[2..],
                )
            })
    }

    pub fn ref_decoration(&self, id: u32, which: Decoration) -> Option<&'_ [Operand]> {
        self.ref_decorations(id)
            .filter_map(|it| if it.0 == which { Some(it.1) } else { None })
            .next()
    }

    pub fn has_ref_decoration(&self, id: u32, with: Decoration) -> bool {
        self.ref_decoration(id, with).is_some()
    }

    pub fn ref_name(&self, id: u32) -> Option<&str> {
        self.debug_names
            .iter()
            .filter(|it| it.class.opcode == Op::Name)
            .find(|it| {
                it.operands
                    .first()
                    .map(|it| matches!(it, Operand::IdRef(it) if *it == id))
                    .unwrap_or_default()
            })
            .and_then(|it| match it.operands.get(1) {
                Some(Operand::LiteralString(name)) => Some(name.as_str()),
                _ => None,
            })
    }

    pub fn ref_struct_member_names(&self, struct_id: u32) -> impl Iterator<Item = (usize, &'_ str)> + '_ {
        self.debug_names
            .iter()
            .filter(|it| it.class.opcode == Op::MemberName)
            .filter(move |it| {
                it.operands
                    .first()
                    .map(|it| matches!(it, Operand::IdRef(it) if *it == struct_id))
                    .unwrap_or_default()
            })
            .map(|it| (
                match it.operands.get(1) {
                    Some(Operand::LiteralBit32(i)) => *i as usize,
                    _ => unreachable!("expected OpMemberName second argument to be member index (LiteralBit32)")
                },
                match it.operands.get(2) {
                    Some(Operand::LiteralString(name)) => name.as_str(),
                    _ => unreachable!("expected OpMemberName third argument to be member name (LiteralString)"),
                }
            ))
    }

    pub fn ref_struct_member_decorations(
        &self,
        struct_id: u32,
    ) -> impl Iterator<Item = (usize, Decoration, &'_ [Operand])> + '_ {
        self.annotations
            .iter()
            .filter(move |it| {
                it.class.opcode == Op::MemberDecorate
                    && it
                        .operands
                        .first()
                        .map(|it| *it == Operand::IdRef(struct_id))
                        .unwrap_or_default()
            })
            .map(|it| {
                (
                    it.operands
                        .get(1)
                        .and_then(|it| match it {
                            Operand::LiteralBit32(field) => Some(*field),
                            _ => None,
                        })
                        .unwrap() as usize,
                    it.operands
                        .get(2)
                        .and_then(|it| match it {
                            Operand::Decoration(it) => Some(*it),
                            _ => None,
                        })
                        .unwrap(),
                    &it.operands[3..],
                )
            })
    }

    pub fn ref_instruction(&self, id: u32) -> Option<&Instruction> {
        self.all_inst_iter().find(|it| it.result_id == Some(id))
    }
    
    pub fn ref_operation(&self, id: u32) -> Option<(Op, &[Operand])> {
        self.ref_instruction(id).map(|it| {
            (it.class.opcode, it.operands.as_slice())
        })
    }

    pub fn ref_constant_value(&self, id: u32) -> Option<SpirvConstantValue> {
        let instruction = self.ref_instruction(id)?;
        SpirvConstantValue::from_instruction(self, instruction, None)
    }

    pub fn ref_value(&self, id: u32) -> Option<SpirvValue> {
        let instruction = self.ref_instruction(id)?;
        SpirvValue::from_instruction(self, instruction, None)
    }

    pub fn ref_storage_class(&self, id: u32) -> Option<StorageClass> {
        let it = self
            .types_global_values
            .iter()
            .find(move |it| it.result_id == Some(id))?;
        match it.class.opcode {
            Op::Variable => match_operand!(it.operands, 0, Operand::StorageClass),
            Op::TypePointer => match_operand!(it.operands, 0, Operand::StorageClass),
            _ => None,
        }
    }

    pub fn descriptor_info_for(&self, id: u32, storage_class: StorageClass) -> Option<DescriptorInfo> {
        let instruction: &Instruction = self.ref_instruction(id)?;
        let instruction = if instruction.class.opcode == Op::Variable {
            let variable_type = instruction
                .result_type
                .expect("missing variable result type");
            assert_eq!(
                Some(storage_class),
                match_operand!(instruction.operands, 0, Operand::StorageClass)
            );
            self.ref_instruction(variable_type)?
        } else {
            instruction
        };
        let id = instruction.result_id.expect("type has no result id");

        match instruction.class.opcode {
            spirv::Op::TypeArray => {
                let element_type_id = match_operand!(instruction.operands, 0, Operand::IdRef)?;
                let num_elements_id = match_operand!(instruction.operands, 1, Operand::IdRef)?;
                let num_elements = self.ref_instruction(num_elements_id)?;
                assert_eq!(num_elements.class.opcode, spirv::Op::Constant);
                let num_elements_ty =
                    self.ref_instruction(num_elements.result_type.expect("no array count result"))?;
                // Array size can be any width, any signedness
                assert_eq!(num_elements_ty.class.opcode, spirv::Op::TypeInt);
                let num_elements =
                    match match_operand!(num_elements_ty.operands, 0, Operand::LiteralBit32)? {
                        32 => match_operand!(num_elements.operands, 0, Operand::LiteralBit32)?
                            as usize,
                        64 => match_operand!(num_elements.operands, 0, Operand::LiteralBit64)?
                            as usize,
                        other => todo!("int width not implemented: {}", other),
                    };
                assert!(num_elements >= 1);
                return Some(DescriptorInfo {
                    binding_count: Some(num_elements),
                    ..self.descriptor_info_for(element_type_id, storage_class)?
                });
            }
            spirv::Op::TypeRuntimeArray => {
                let element_type_id = match_operand!(instruction.operands, 0, Operand::IdRef)?;
                return Some(DescriptorInfo {
                    binding_count: None,
                    ..self.descriptor_info_for(element_type_id, storage_class)?
                });
            }
            spirv::Op::TypePointer => {
                assert_eq!(
                    storage_class,
                    match_operand!(instruction.operands, 0, Operand::StorageClass)?,
                    "variable storage class must match corresponding type pointer class"
                );
                let element_type_id = match_operand!(instruction.operands, 1, Operand::IdRef)?;
                return self.descriptor_info_for(element_type_id, storage_class);
            }
            spirv::Op::TypeSampledImage => {
                let element_type_id = match_operand!(instruction.operands, 0, Operand::IdRef)?;

                let image_instruction = self.ref_instruction(element_type_id)?;
                let descriptor = image_instruction
                    .result_id
                    .and_then(|id| self.descriptor_info_for(id, storage_class))?;

                let dim = match_operand!(image_instruction.operands, 1, Operand::Dim)?;
                assert_ne!(dim, spirv::Dim::DimSubpassData);

                return match (dim, descriptor.ty) {
                    (spirv::Dim::DimBuffer, DescriptorType::UniformTexelBuffer) => Some(descriptor),
                    (spirv::Dim::DimBuffer, DescriptorType::StorageTexelBuffer) => Some(descriptor),
                    (spirv::Dim::DimBuffer, _) => {
                        todo!("unhandled sampled image type {:?}", descriptor.ty)
                    }
                    _ => Some(DescriptorInfo {
                        ty: DescriptorType::CombinedImageSampler,
                        ..descriptor
                    }),
                };
            }
            _ => {}
        }

        let version = self.version();
        let descriptor_type = match instruction.class.opcode {
            spirv::Op::TypeSampler => DescriptorType::Sampler,
            spirv::Op::TypeImage => {
                let dim = match_operand!(instruction.operands, 1, Operand::Dim)?;
                let sampled = ImageSampling::from_literal(match_operand!(
                    instruction.operands,
                    5,
                    Operand::LiteralBit32
                )?);

                match (dim, sampled) {
                    (spirv::Dim::DimBuffer, ImageSampling::Dynamic) => {
                        unimplemented!("either UniformTexelBuffer or StorageTexelBuffer")
                    }
                    (spirv::Dim::DimBuffer, ImageSampling::Sampled) => {
                        DescriptorType::UniformTexelBuffer
                    }
                    (spirv::Dim::DimBuffer, ImageSampling::Storage) => {
                        DescriptorType::StorageTexelBuffer
                    }
                    (spirv::Dim::DimSubpassData, _) => DescriptorType::InputAttachment,
                    (_, ImageSampling::Sampled) => DescriptorType::SampledImage,
                    (_, ImageSampling::Storage) => DescriptorType::StorageImage,
                    (dim, sampling) => todo!(
                        "image type with dim={:?} and sampling={:?} not supported",
                        dim,
                        sampling
                    ),
                }
            }
            spirv::Op::TypeStruct => {
                let has_block = self.has_ref_decoration(id, Decoration::Block);
                let has_buffer_block = !has_block
                    && version <= (1, 3)
                    && self.has_ref_decoration(id, Decoration::BufferBlock);

                if !has_block && !has_buffer_block {
                    if version >= (1, 3) {
                        panic!("invalid TypeStruct descriptor: no `Block` decoration")
                    } else {
                        panic!(
                            "invalid TypeStruct descriptor: no `Block` or `BufferBlock` decoration"
                        )
                    }
                }

                if has_buffer_block {
                    DescriptorType::StorageBuffer
                } else if version < (1, 3) {
                    DescriptorType::UniformBuffer
                } else {
                    match storage_class {
                        spirv::StorageClass::StorageBuffer => DescriptorType::StorageBuffer,
                        spirv::StorageClass::Uniform | spirv::StorageClass::UniformConstant => {
                            DescriptorType::UniformBuffer
                        }
                        other => todo!("unknown block storage class: {:?}", other),
                    }
                }
            }
            spirv::Op::TypeAccelerationStructureKHR => DescriptorType::AccelerationStructureKhr,
            other => todo!("type not implemented: {:?}", other),
        };

        Some(DescriptorInfo {
            binding_count: Some(1),
            ty: descriptor_type,
        })
    }

    pub fn ref_type(&self, id: u32) -> Option<&SpirvTypeOwned> {
        if let Some(cached) = self.types.get(&id) {
            return Some(cached)
        }

        let instruction = self
            .types_global_values
            .iter()
            .find(|it| it.result_id == Some(id))
            .expect("invalid instruction id reference");
        self.instruction_type(instruction)
    }

    pub fn instruction_type(&self, declaration: &Instruction) -> Option<&SpirvTypeOwned> {
        if declaration.class.opcode == Op::Variable {
            let variable_type = declaration
                .result_type
                .expect("missing variable result type");
            return self.ref_type(variable_type);
        }

        let id = declaration
            .result_id
            .expect("type instruction must have a result id");
        self.types.get(&id)
    }
}

pub struct DescriptorInfo {
    pub binding_count: Option<usize>,
    pub ty: DescriptorType,
}
