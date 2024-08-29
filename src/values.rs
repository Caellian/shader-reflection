use rspirv::dr::{Instruction, Operand};
use spirv::*;
use serde::{Serialize, Deserialize};

use crate::{types::*, ext::ModuleExt};

macro_rules! match_operand {
    ($source: expr, $index: literal, $operand: path) => {
        match $source.get($index) {
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

macro_rules! unwrap_instruction_type {
    ($value: ty, $inner: ident) => {
        impl FromInstruction<SpirvTypeOwned> for $value {
            fn from_instruction(
                module: &ModuleExt,
                instruction: &Instruction,
                ty: Option<&SpirvTypeOwned>,
            ) -> Option<Self> {
                match ty {
                    Some(SpirvTypeOwned::$inner(it)) => {
                        Self::from_instruction(module, instruction, Some(it))
                    }
                    _ => None,
                }
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SprivConstantInteger {
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
}
impl FromInstruction<SpirvTypeOwned> for SprivConstantInteger {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvTypeOwned>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                instruction.result_type
                    .and_then(|id| module.ref_type(id))
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::Constant {
            return None;
        }

        match ty {
            SpirvTypeOwned::Literal(SpirvLiteralType::Integer {
                width: 32,
                signed: false,
                ..
            }) => {
                let value = expect_operand!(
                    instruction.operands,
                    0,
                    LiteralBit32,
                    "32-bit unsigned integer value"
                );
                Some(SprivConstantInteger::U32(value))
            }
            SpirvTypeOwned::Literal(SpirvLiteralType::Integer {
                width: 32,
                signed: true,
                ..
            }) => {
                let value = expect_operand!(
                    instruction.operands,
                    0,
                    LiteralBit32,
                    "32-bit signed integer value"
                );
                Some(SprivConstantInteger::I32(i32::from_le_bytes(
                    value.to_le_bytes(),
                )))
            }
            SpirvTypeOwned::Literal(SpirvLiteralType::Integer {
                width: 64,
                signed: false,
                ..
            }) => {
                let value = expect_operand!(
                    instruction.operands,
                    0,
                    LiteralBit64,
                    "64-bit unsigned integer value"
                );
                Some(SprivConstantInteger::U64(value))
            }
            SpirvTypeOwned::Literal(SpirvLiteralType::Integer {
                width: 64,
                signed: true,
                ..
            }) => {
                let value = expect_operand!(
                    instruction.operands,
                    0,
                    LiteralBit64,
                    "64-bit signed integer value"
                );
                Some(SprivConstantInteger::I64(i64::from_le_bytes(
                    value.to_le_bytes(),
                )))
            }
            SpirvTypeOwned::Literal(ty) if matches!(ty, SpirvLiteralType::Integer { .. }) => {
                todo!("unsupported constant integer type: {:?}", ty)
            }
            _ => None,
        }
    }
}
impl TryInto<i32> for SprivConstantInteger {
    type Error = ();

    fn try_into(self) -> Result<i32, Self::Error> {
        match self {
            SprivConstantInteger::I32(it) => Ok(it),
            SprivConstantInteger::U32(it) => {
                if it > i32::MAX as u32 {
                    return Err(());
                }
                Ok(it as i32)
            }
            SprivConstantInteger::I64(it) => {
                if it.is_negative() || it > i32::MAX as i64 {
                    return Err(());
                }
                Ok(it as i32)
            }
            SprivConstantInteger::U64(it) => {
                if it > i32::MAX as u64 {
                    return Err(());
                }
                Ok(it as i32)
            }
        }
    }
}
impl TryInto<u32> for SprivConstantInteger {
    type Error = ();

    fn try_into(self) -> Result<u32, Self::Error> {
        match self {
            SprivConstantInteger::I32(it) => {
                if it.is_negative() {
                    return Err(());
                }
                Ok(it as u32)
            }
            SprivConstantInteger::U32(it) => Ok(it),
            SprivConstantInteger::I64(it) => {
                if it.is_negative() || it > u32::MAX as i64 {
                    return Err(());
                }
                Ok(it as u32)
            }
            SprivConstantInteger::U64(it) => {
                if it > u32::MAX as u64 {
                    return Err(());
                }
                Ok(it as u32)
            }
        }
    }
}
impl TryInto<i64> for SprivConstantInteger {
    type Error = ();

    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            SprivConstantInteger::I32(it) => Ok(it as i64),
            SprivConstantInteger::U32(it) => Ok(it as i64),
            SprivConstantInteger::I64(it) => Ok(it),
            SprivConstantInteger::U64(it) => {
                if it > i64::MAX as u64 {
                    return Err(());
                }
                Ok(it as i64)
            }
        }
    }
}
impl TryInto<u64> for SprivConstantInteger {
    type Error = ();

    fn try_into(self) -> Result<u64, Self::Error> {
        match self {
            SprivConstantInteger::I32(it) => {
                if it.is_negative() {
                    return Err(());
                }
                Ok(it as u64)
            }
            SprivConstantInteger::U32(it) => Ok(it as u64),
            SprivConstantInteger::I64(it) => {
                if it.is_negative() {
                    return Err(());
                }
                Ok(it as u64)
            }
            SprivConstantInteger::U64(it) => Ok(it),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SprivConstantFloat {
    F32(f32),
    F64(f64),
}
impl FromInstruction<SpirvTypeOwned> for SprivConstantFloat {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvTypeOwned>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                match_operand!(instruction.operands, 0, Operand::IdRef)
                    .and_then(|id| module.ref_type(id))
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::Constant {
            return None;
        }

        match ty {
            SpirvTypeOwned::Literal(SpirvLiteralType::Float {
                width: 32,
                encoding: None,
                ..
            }) => {
                let value =
                    expect_operand!(instruction.operands, 1, LiteralBit32, "32-bit float value");
                Some(SprivConstantFloat::F32(f32::from_le_bytes(
                    value.to_le_bytes(),
                )))
            }
            SpirvTypeOwned::Literal(SpirvLiteralType::Float {
                width: 64,
                encoding: None,
                ..
            }) => {
                let value =
                    expect_operand!(instruction.operands, 1, LiteralBit64, "64-bit float value");
                Some(SprivConstantFloat::F64(f64::from_le_bytes(
                    value.to_le_bytes(),
                )))
            }
            SpirvTypeOwned::Literal(ty) if matches!(ty, SpirvLiteralType::Float { .. }) => {
                todo!("unsupported constant float type: {:?}", ty)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SprivConstantArray {
    values: Vec<SpirvConstantValue>,
}

unwrap_instruction_type!(SprivConstantArray, Array);
impl FromInstruction<SpirvArrayType> for SprivConstantArray {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvArrayType>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                let ty = match_operand!(instruction.operands, 0, Operand::IdRef)
                    .and_then(|id| module.ref_type(id));
                match ty {
                    Some(SpirvTypeOwned::Array(it)) => Some(it),
                    _ => None,
                }
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::ConstantComposite {
            return None;
        }

        let length = ty
            .length
            .expect("constant value array can't have variable length");
        let values = instruction.operands[1..=length]
            .iter()
            .map(|it| match it {
                Operand::IdRef(id) => {
                    let instruction = module
                        .ref_instruction(*id)
                        .expect("unable to find constant array value instruction");
                    SpirvConstantValue::from_instruction(
                        module,
                        instruction,
                        Some(ty.value_type.as_ref()),
                    )
                    .expect("unable to parse array value")
                }
                _ => unreachable!("array values must be id references"),
            })
            .collect();

        Some(SprivConstantArray { values })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SprivConstantVector {
    values: Vec<SpirvConstantValue>,
}

unwrap_instruction_type!(SprivConstantVector, Vector);
impl FromInstruction<SpirvVectorType> for SprivConstantVector {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvVectorType>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                let ty = match_operand!(instruction.operands, 0, Operand::IdRef)
                    .and_then(|id| module.ref_type(id));
                match ty {
                    Some(SpirvTypeOwned::Vector(vec_ty)) => Some(vec_ty),
                    _ => None,
                }
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::ConstantComposite {
            return None;
        }

        let values = instruction.operands[1..=ty.width]
            .iter()
            .map(|it| match it {
                Operand::IdRef(id) => {
                    let instruction = module
                        .ref_instruction(*id)
                        .expect("unable to find constant vector value instruction");
                    SpirvConstantValue::from_instruction(
                        module,
                        instruction,
                        Some(ty.component_type.as_ref()),
                    )
                    .expect("unable to parse vector value")
                }
                _ => unreachable!("vector values must be id references"),
            })
            .collect();

        Some(SprivConstantVector { values })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SprivConstantMatrix {
    values: Vec<SprivConstantVector>,
}

unwrap_instruction_type!(SprivConstantMatrix, Matrix);
impl FromInstruction<SpirvMatrixType> for SprivConstantMatrix {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvMatrixType>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                let ty = match_operand!(instruction.operands, 0, Operand::IdRef)
                    .and_then(|id| module.ref_type(id));
                match ty {
                    Some(SpirvTypeOwned::Matrix(it)) => Some(it),
                    _ => None,
                }
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::ConstantComposite {
            return None;
        }

        let values = instruction
            .operands
            .iter()
            .skip(1)
            .take(ty.columns)
            .map(|it| match it {
                Operand::IdRef(id) => {
                    let instruction = module
                        .ref_instruction(*id)
                        .expect("unable to find constant matrix vector instruction");
                    SprivConstantVector::from_instruction(
                        module,
                        instruction,
                        Some(&ty.column_type),
                    )
                    .expect("unable to parse matrix vector")
                }
                _ => unreachable!("matrix values must be id references"),
            })
            .collect();

        Some(SprivConstantMatrix { values })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SprivConstantStruct {
    values: Vec<SpirvConstantValue>,
}

unwrap_instruction_type!(SprivConstantStruct, Struct);
impl FromInstruction<SpirvStructType> for SprivConstantStruct {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvStructType>,
    ) -> Option<Self> {
        let ty = context
            .or_else(|| {
                let ty = match_operand!(instruction.operands, 0, Operand::IdRef)
                    .and_then(|id| module.ref_type(id));
                match ty {
                    Some(SpirvTypeOwned::Struct(struct_ty)) => Some(struct_ty),
                    _ => None,
                }
            })
            .expect("constant type not specified");

        if unspecialize(instruction.class.opcode) != Op::ConstantComposite {
            return None;
        }

        let values = instruction
            .operands
            .iter()
            .skip(1)
            .map(|it| match it {
                Operand::IdRef(id) => module
                    .ref_instruction(*id)
                    .expect("unable to find constant struct field instruction"),
                _ => unreachable!("struct fields must be id references"),
            })
            .zip(ty.fields.iter().map(|it| &it.ty))
            .map(|(instruction, ty)| {
                SpirvConstantValue::from_instruction(module, instruction, Some(ty))
                    .expect("unable to parse struct field")
            })
            .collect();

        Some(SprivConstantStruct { values })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstantVariant {
    Bool(bool),
    Integer(SprivConstantInteger),
    Float(SprivConstantFloat),
    Array(SprivConstantArray),
    Vector(SprivConstantVector),
    Matrix(SprivConstantMatrix),
    Struct(SprivConstantStruct),
    Null,
}

impl FromInstruction<SpirvTypeOwned> for ConstantVariant {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&SpirvTypeOwned>,
    ) -> Option<Self> {
        let ty = context.expect("constant type not specified");
        match (unspecialize(instruction.class.opcode), ty) {
            (Op::ConstantTrue, _) => Some(ConstantVariant::Bool(true)),
            (Op::ConstantFalse, _) => Some(ConstantVariant::Bool(false)),
            // Majority of modern consumer-grade GPUs are little-endian.
            (Op::Constant, SpirvTypeOwned::Literal(SpirvLiteralType::Integer { .. })) => {
                SprivConstantInteger::from_instruction(module, instruction, Some(ty))
                    .map(ConstantVariant::Integer)
            }
            (Op::Constant, SpirvTypeOwned::Literal(SpirvLiteralType::Float { .. })) => {
                SprivConstantFloat::from_instruction(module, instruction, Some(ty))
                    .map(ConstantVariant::Float)
            }
            (Op::ConstantComposite, SpirvTypeOwned::Array(array_ty)) => {
                SprivConstantArray::from_instruction(module, instruction, Some(array_ty))
                    .map(ConstantVariant::Array)
            }
            (Op::ConstantComposite, SpirvTypeOwned::Vector(vec_ty)) => {
                SprivConstantVector::from_instruction(module, instruction, Some(vec_ty))
                    .map(ConstantVariant::Vector)
            }
            (Op::ConstantComposite, SpirvTypeOwned::Matrix(mat_ty)) => {
                SprivConstantMatrix::from_instruction(module, instruction, Some(mat_ty))
                    .map(ConstantVariant::Matrix)
            }
            (Op::ConstantComposite, SpirvTypeOwned::Struct(struct_ty)) => {
                SprivConstantStruct::from_instruction(module, instruction, Some(struct_ty))
                    .map(ConstantVariant::Struct)
            }
            (Op::ConstantNull, _) => Some(ConstantVariant::Null),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpirvConstantValue {
    ty: SpirvTypeOwned,
    variant: ConstantVariant,
    can_specialize: bool,
}

impl FromInstruction<SpirvTypeOwned> for SpirvConstantValue {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        ty: Option<&SpirvTypeOwned>,
    ) -> Option<Self> {
        let ty = ty.unwrap_or_else(|| {
            module
                .ref_type(
                    instruction
                        .result_type
                        .expect("missing constant type reference"),
                )
                .expect("unable to resolve constant type")
        });
        let variant = ConstantVariant::from_instruction(module, instruction, Some(&ty))?;
        let can_specialize = matches!(instruction.class.opcode,
            Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
            | Op::SpecConstantCompositeContinuedINTEL
        );
        Some(SpirvConstantValue {
            ty: ty.clone(),
            variant,
            can_specialize,
        })
    }
}

fn unspecialize(op: Op) -> Op {
    match op {
        Op::SpecConstantTrue => Op::ConstantTrue,
        Op::SpecConstantFalse => Op::ConstantFalse,
        Op::SpecConstant => Op::Constant,
        Op::SpecConstantComposite => Op::ConstantComposite,
        Op::SpecConstantCompositeContinuedINTEL => Op::ConstantCompositeContinuedINTEL,
        other => other,
    }
}

impl std::ops::Deref for SpirvConstantValue {
    type Target = ConstantVariant;

    fn deref(&self) -> &Self::Target {
        &self.variant
    }
}

pub enum SpirvValue {
    Constant(SpirvConstantValue),
    Variable {
        ty: SpirvTypeOwned,
        reference: u32,
        default: Option<SpirvConstantValue>,
    },
}

impl FromInstruction for SpirvValue {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        todo!() // FIXME
    }
}
