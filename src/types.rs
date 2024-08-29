use std::ops::Deref;

use rspirv::dr::{Instruction, Module, Operand};
use serde::{Deserialize, Serialize};
use spirv::{Decoration, Dim, ImageFormat, Op, StorageClass};

use crate::{ext::ModuleExt, ImageSampling, values::SprivConstantInteger};

/*
TODO: Types should actually store references to other types, all owned by module. But this requires:
 - Custom deserializer for Module wrapper because individual types need to reference eachother so
   deserializer needs Module as argument to connect them properly.
 - Custom accessors which override some values (e.g. array stride) based on SPIR-V Descriptors.

That's a week of work to achieve smaller memory footprint at cost of much more complicated accessing
and deserialization logic.
*/

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

pub trait FromInstruction<T = ()>: Sized {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        context: Option<&T>,
    ) -> Option<Self>;
    fn from_instruction_ref(module: &ModuleExt, id: u32, context: Option<&T>) -> Option<Self> {
        let instruction = module.ref_instruction(id)?;
        Self::from_instruction(module, instruction, context)
    }
}

pub trait SpirvType {
    fn type_reference(&self) -> u32;

    fn as_typed_ref(&self) -> SpirvTypeRef<'_>;
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_>;

    fn memory_size_bits(&self) -> usize {
        0
    }
    fn memory_size(&self) -> usize {
        self.memory_size_bits().div_ceil(8)
    }
    fn memory_alignment(&self) -> usize {
        1
    }
}
pub trait SpirvUniformCollection: SpirvType {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn element_ty(&self) -> &dyn SpirvType;
    fn element_ty_mut(&mut self) -> &mut dyn SpirvType;
}

macro_rules! no_element {
    (ref $name: ident) => {
        $name
    };
    (mut ref $name: ident) => {
        $name
    };
}
macro_rules! has_element {
    (ref $name: ident) => {
        $name.element_ty()
    };
    (mut ref $name: ident) => {
        $name.element_ty_mut()
    };
}
macro_rules! base_type {
    ($($name: ident: $element_access: ident, ops: $($op_code: ident)|+);+ $(;)?) => {paste::paste!{
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum SpirvTypeRef<'a> {
            $($name(&'a [<Spirv $name Type>])),*
        }
        impl<'a> SpirvTypeRef<'a> {
            fn to_owned(&self) -> SpirvTypeOwned {
                match *self {
                    $(SpirvTypeRef::$name(it) => SpirvTypeOwned::$name(it.clone())),*
                }
            }
        }

        #[derive(Debug, PartialEq, Eq)]
        pub enum SpirvTypeRefMut<'a> {
            $($name(&'a mut [<Spirv $name Type>])),*
        }
        impl<'a> SpirvTypeRefMut<'a> {
            fn to_owned(&mut self) -> SpirvTypeOwned {
                match self {
                    $(SpirvTypeRefMut::$name(it) => SpirvTypeOwned::$name(it.clone())),*
                }
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub enum SpirvTypeOwned {
            $($name([<Spirv $name Type>])),*
        }
        impl SpirvTypeOwned {
            pub fn element(&self) -> &dyn SpirvType {
                match self {
                    $(SpirvTypeOwned::$name(it) => $element_access!(ref it)),*
                }
            }
            pub fn element_mut(&mut self) -> &mut dyn SpirvType {
                match self {
                    $(SpirvTypeOwned::$name(it) => $element_access!(mut ref it)),*
                }
            }
        }
        impl SpirvType for SpirvTypeOwned {
            fn type_reference(&self) -> u32 {
                match self {
                    $(SpirvTypeOwned::$name(it) => it.type_reference()),*
                }
            }
            fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
                match self {
                    $(SpirvTypeOwned::$name(it) => SpirvTypeRef::$name(it)),*
                }
            }
            fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
                match self {
                    $(SpirvTypeOwned::$name(it) => SpirvTypeRefMut::$name(it)),*
                }
            }
            fn memory_size_bits(&self) -> usize {
                match self {
                    $(SpirvTypeOwned::$name(it) => it.memory_size_bits()),*
                }
            }
            fn memory_alignment(&self) -> usize {
                match self {
                    $(SpirvTypeOwned::$name(it) => it.memory_alignment()),*
                }
            }
        }
        impl FromInstruction for SpirvTypeOwned {
            fn from_instruction(module: &ModuleExt, instruction: &Instruction, _: Option<&()>) -> Option<Self> {
                Some(match instruction.class.opcode {
                    $(
                        $(Op::[<Type $op_code>])|+ => {
                            SpirvTypeOwned::$name([<Spirv $name Type>]::from_instruction(module, instruction, None)?)
                        }
                    ),*
                    _ => return None
                })
            }
        }
    }};
}

base_type![
    Literal: no_element, ops: Void | Bool | Int | Float;
    Array: has_element, ops: Array | RuntimeArray;
    Vector: has_element, ops: Vector;
    Matrix: has_element, ops: Matrix;
    Struct: no_element, ops: Struct;
    Pointer: has_element, ops: Pointer;
    SampledImage: no_element, ops: SampledImage;
    Image: no_element, ops: Image;
];

impl Default for SpirvTypeOwned {
    fn default() -> Self {
        SpirvTypeOwned::Literal(SpirvLiteralType::Void { type_reference: 0 })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpirvLiteralType {
    Float {
        width: u32,
        encoding: Option<()>,
        type_reference: u32,
    },
    Integer {
        width: u32,
        signed: bool,
        type_reference: u32,
    },
    Boolean {
        type_reference: u32,
    },
    Void {
        type_reference: u32,
    },
}
impl SpirvType for SpirvLiteralType {
    fn type_reference(&self) -> u32 {
        match self {
            SpirvLiteralType::Float { type_reference, .. }
            | SpirvLiteralType::Integer { type_reference, .. }
            | SpirvLiteralType::Boolean { type_reference }
            | SpirvLiteralType::Void { type_reference } => *type_reference,
        }
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Literal(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Literal(self)
    }
    fn memory_size_bits(&self) -> usize {
        match self {
            SpirvLiteralType::Float { width, .. } => *width as usize,
            SpirvLiteralType::Integer { width, .. } => *width as usize,
            SpirvLiteralType::Boolean { .. } => 0, // not defined (internal)
            SpirvLiteralType::Void { .. } => 0,    // not defined (internal)
        }
    }
    fn memory_alignment(&self) -> usize {
        match self {
            SpirvLiteralType::Float { width, .. } => (width.div_ceil(8)) as usize,
            SpirvLiteralType::Integer { width, .. } => (width.div_ceil(8)) as usize,
            SpirvLiteralType::Boolean { .. } => 1,
            SpirvLiteralType::Void { .. } => 1,
        }
    }
}
impl FromInstruction for SpirvLiteralType {
    fn from_instruction(_: &ModuleExt, instruction: &Instruction, _: Option<&()>) -> Option<Self> {
        let id = instruction.result_id?;

        Some(match instruction.class.opcode {
            Op::TypeVoid => SpirvLiteralType::Void { type_reference: id },
            Op::TypeBool => SpirvLiteralType::Boolean { type_reference: id },
            Op::TypeInt => {
                let width = expect_operand!(instruction.operands, 0, LiteralBit32, "width");
                let signedness =
                    expect_operand!(instruction.operands, 1, LiteralBit32, "signedness");
                SpirvLiteralType::Integer {
                    width,
                    signed: signedness == 1,
                    type_reference: id,
                }
            }
            Op::TypeFloat => {
                let width = expect_operand!(instruction.operands, 0, LiteralBit32, "width");
                let encoding = instruction.operands.len() > 1;
                SpirvLiteralType::Float {
                    width,
                    encoding: if encoding {
                        todo!("custom OpTypeFloat encoding not yet implemented")
                    } else {
                        None
                    },
                    type_reference: id,
                }
            }
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvArrayType {
    pub(crate) length: Option<usize>,
    pub(crate) stride: Option<usize>,
    pub(crate) value_type: Box<SpirvTypeOwned>,
    pub(crate) type_reference: u32,
}
impl SpirvType for SpirvArrayType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Array(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Array(self)
    }
    fn memory_size_bits(&self) -> usize {
        self.length
            .map(|length| length * self.value_type.memory_size_bits())
            .unwrap_or_default()
    }
    fn memory_alignment(&self) -> usize {
        self.value_type.memory_alignment()
    }
}
impl SpirvUniformCollection for SpirvArrayType {
    fn len(&self) -> usize {
        self.length.unwrap_or_default()
    }
    fn element_ty(&self) -> &dyn SpirvType {
        match self.value_type.as_ref() {
            SpirvTypeOwned::Array(inner) => inner.element_ty(),
            other => other,
        }
    }
    fn element_ty_mut(&mut self) -> &mut dyn SpirvType {
        match self.value_type.as_mut() {
            SpirvTypeOwned::Array(inner) => inner.element_ty_mut(),
            other => other,
        }
    }
}
impl FromInstruction for SpirvArrayType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        if instruction.class.opcode != Op::TypeArray
            && instruction.class.opcode != Op::TypeRuntimeArray
        {
            return None;
        }

        let id = instruction.result_id?;
        let element_type = expect_operand!(instruction.operands, 0, IdRef, "element type");
        let mut element_type = module
            .types
            .get(&element_type)
            .expect("can't get array element type")
            .clone();
        let length = if instruction.class.opcode == Op::TypeArray {
            let length = expect_operand!(instruction.operands, 1, IdRef, "length");
            let length = SprivConstantInteger::from_instruction_ref(module, length, None)
                .expect("can't parse constant length");
            let length: u32 = length.try_into().expect("invalid length value");
            assert!(length >= 1, "array length must must be at least 1");
            Some(length as usize)
        } else {
            None
        };
        let mut stride = None;
        for (decoration, args) in module.ref_decorations(id) {
            let element_type_ref = element_type.element_mut().as_typed_ref_mut();
            match (element_type_ref, decoration) {
                (_, Decoration::ArrayStride) => {
                    stride = Some(expect_operand!(args, 0, LiteralBit32, "array stride") as usize)
                }
                (SpirvTypeRefMut::Matrix(matrix), Decoration::MatrixStride) => {
                    matrix.stride =
                        Some(expect_operand!(args, 0, LiteralBit32, "matrix stride") as usize);
                }
                (SpirvTypeRefMut::Matrix(matrix), Decoration::RowMajor) => {
                    assert_ne!(
                        matrix.row_major,
                        Some(false),
                        "array element matrix layout has already been set to column major"
                    );
                    matrix.row_major = Some(true);
                }
                (SpirvTypeRefMut::Matrix(matrix), Decoration::ColMajor) => {
                    assert_ne!(
                        matrix.row_major,
                        Some(true),
                        "array element matrix layout has already been set to row major"
                    );
                    matrix.row_major = Some(false);
                }
                (element_type, other) => todo!(
                    "unhandled decoration {:?} for array with element type: {:?}",
                    other,
                    element_type
                ),
            }
        }
        Some(SpirvArrayType {
            length,
            stride,
            value_type: Box::new(element_type.clone()),
            type_reference: id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvVectorType {
    pub(crate) width: usize,
    pub(crate) component_type: Box<SpirvTypeOwned>,
    pub(crate) type_reference: u32,
}
impl SpirvVectorType {
    pub fn to_array(&self) -> SpirvArrayType {
        SpirvArrayType {
            length: Some(self.width),
            stride: None,
            value_type: self.component_type.clone(),
            type_reference: self.type_reference,
        }
    }
}
impl SpirvType for SpirvVectorType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Vector(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Vector(self)
    }
    fn memory_size_bits(&self) -> usize {
        self.width * self.component_type.memory_size_bits()
    }
    fn memory_alignment(&self) -> usize {
        self.component_type.memory_alignment()
    }
}
impl SpirvUniformCollection for SpirvVectorType {
    fn len(&self) -> usize {
        self.width
    }
    fn element_ty(&self) -> &dyn SpirvType {
        self.component_type.as_ref()
    }
    fn element_ty_mut(&mut self) -> &mut dyn SpirvType {
        self.component_type.as_mut()
    }
}
impl FromInstruction for SpirvVectorType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        if instruction.class.opcode != Op::TypeVector {
            return None;
        }

        let id = instruction.result_id?;
        let component_type = expect_operand!(instruction.operands, 0, IdRef, "component type");
        let component_type = module
            .types
            .get(&component_type)
            .expect("can't get vector component type");
        let width = expect_operand!(instruction.operands, 1, LiteralBit32, "width") as usize;
        assert!(width >= 2, "vector must have at least 2 components");
        Some(SpirvVectorType {
            width,
            component_type: Box::new(component_type.clone()),
            type_reference: id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvMatrixType {
    pub(crate) columns: usize,
    pub(crate) stride: Option<usize>,
    pub(crate) row_major: Option<bool>,
    pub(crate) column_type: SpirvVectorType,
    pub(crate) type_reference: u32,
}
impl SpirvMatrixType {
    pub fn is_row_major(&self) -> bool {
        self.row_major.unwrap_or_default()
    }
}
impl SpirvType for SpirvMatrixType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Matrix(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Matrix(self)
    }
    fn memory_size_bits(&self) -> usize {
        self.columns
            * self
                .stride
                .unwrap_or_else(|| self.column_type.memory_size_bits())
    }
    fn memory_alignment(&self) -> usize {
        self.element_ty().memory_alignment()
    }
}
impl SpirvUniformCollection for SpirvMatrixType {
    fn len(&self) -> usize {
        self.columns * self.column_type.len()
    }
    fn element_ty(&self) -> &dyn SpirvType {
        self.column_type.element_ty()
    }
    fn element_ty_mut(&mut self) -> &mut dyn SpirvType {
        self.column_type.element_ty_mut()
    }
}
impl FromInstruction for SpirvMatrixType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        if instruction.class.opcode != Op::TypeMatrix {
            return None;
        }

        let id = instruction.result_id?;
        let column_type = expect_operand!(instruction.operands, 0, IdRef, "column type");
        let column_type = module
            .types
            .get(&column_type)
            .expect("can't get matrix column type");
        let column_type = match column_type {
            SpirvTypeOwned::Vector(it) => it.clone(),
            other => panic!(
                "expected matrix column type to be a vector; found: {:?}",
                other
            ),
        };
        let columns =
            expect_operand!(instruction.operands, 1, LiteralBit32, "column count") as usize;
        assert!(columns >= 2, "matrix must have at least 2 columns");
        let mut stride = None;
        let mut row_major = None;
        for (decoration, args) in module.ref_decorations(id) {
            match decoration {
                Decoration::MatrixStride => {
                    stride = Some(expect_operand!(args, 0, LiteralBit32, "matrix stride") as usize)
                }
                Decoration::RowMajor => {
                    assert_ne!(
                        row_major,
                        Some(false),
                        "matrix layout has already been set to column major"
                    );
                    row_major = Some(true)
                }
                Decoration::ColMajor => {
                    assert_ne!(
                        row_major,
                        Some(true),
                        "matrix layout has already been set to row major"
                    );
                    row_major = Some(false)
                }
                other => todo!("unhandled decoration for array type: {:?}", other),
            }
        }
        Some(SpirvMatrixType {
            columns,
            stride,
            row_major,
            column_type,
            type_reference: id,
        })
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvStructField {
    pub(crate) name: Option<String>,
    pub(crate) offset: usize,
    pub(crate) ty: SpirvTypeOwned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvStructType {
    pub(crate) name: Option<String>,
    pub(crate) fields: Vec<SpirvStructField>,
    pub(crate) interface: bool,
    pub(crate) type_reference: u32,
}
impl SpirvStructType {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn members(&self) -> &[SpirvStructField] {
        &self.fields
    }

    pub fn position_of(&self, field: usize) -> usize {
        self.fields
            .get(field)
            .map(|it| it.offset)
            .unwrap_or_else(|| self.memory_size())
    }

    pub fn padding_before(&self, field: usize) -> Option<usize> {
        if field == 0 {
            let offset = self.fields.first().map(|it| it.offset);
            return match offset {
                Some(0) | None => None,
                other => other,
            };
        }

        let self_pos = self
            .fields
            .get(field)
            .map(|it| it.offset)
            .unwrap_or_else(|| self.position_of(field));
        let prev_end =
            self.position_of(field - 1) + self.fields.get(field - 1).unwrap().ty.memory_size();

        if prev_end == self_pos {
            None
        } else {
            Some(self_pos - prev_end)
        }
    }
}
impl SpirvType for SpirvStructType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Struct(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Struct(self)
    }
    fn memory_size_bits(&self) -> usize {
        self.fields
            .last()
            .map(|last| last.offset * 8 + last.ty.memory_size_bits())
            .unwrap_or_default()
    }
    fn memory_alignment(&self) -> usize {
        self.fields
            .iter()
            .map(|it| it.ty.memory_alignment())
            .max()
            .unwrap_or(1)
    }
}
impl FromInstruction for SpirvStructType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        let id = instruction.result_id?;

        let mut fields: Vec<_> = Vec::new();
        let field_iter = instruction.operands.iter().map(|it| match it {
            Operand::IdRef(it) => module.ref_type(*it).expect("unhandled struct field type"),
            _ => unreachable!("OpTypeStruct can only have IdRef operands"),
        });

        let mut offset = 0usize;
        for ty in field_iter {
            let alignment = ty.memory_alignment();
            offset = offset.div_ceil(alignment) * alignment;
            let size = ty.memory_size_bits();
            fields.push(SpirvStructField {
                ty: ty.clone(),
                offset,
                ..Default::default()
            });
            offset += size;
        }

        for (i, name) in module.ref_struct_member_names(id) {
            let target = match fields.get_mut(i) {
                Some(it) => it,
                None => panic!(
                    "OpMemberName references invalid field index {} of struct {}",
                    i, id
                ),
            };
            target.name = Some(name.to_string());
        }
        for (i, decoration, args) in module.ref_struct_member_decorations(id) {
            let target = match fields.get_mut(i) {
                Some(it) => it,
                None => panic!(
                    "OpMemberDecorate references invalid field index {} of struct {}",
                    i, id
                ),
            };
            match (&mut target.ty, decoration) {
                (_, Decoration::Offset) => {
                    target.offset = expect_operand!(args, 0, LiteralBit32, "field offset") as usize
                }
                (SpirvTypeOwned::Matrix(matrix), Decoration::MatrixStride) => {
                    matrix.stride =
                        Some(expect_operand!(args, 0, LiteralBit32, "matrix stride") as usize);
                }
                (SpirvTypeOwned::Matrix(matrix), Decoration::RowMajor) => {
                    assert_ne!(
                        matrix.row_major,
                        Some(false),
                        "struct field matrix layout has already been set to column major"
                    );
                    matrix.row_major = Some(true);
                }
                (SpirvTypeOwned::Matrix(matrix), Decoration::ColMajor) => {
                    assert_ne!(
                        matrix.row_major,
                        Some(true),
                        "struct field matrix layout has already been set to row major"
                    );
                    matrix.row_major = Some(false);
                }
                (target, other) => todo!(
                    "unhandled struct field ({:?}) decoration: {:?}",
                    target,
                    other
                ),
            }
        }
        let name = module.ref_name(id).map(str::to_string);
        Some(SpirvStructType {
            name,
            fields,
            interface: module.has_ref_decoration(id, Decoration::Block),
            type_reference: id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvPointerType {
    pub(crate) storage_class: StorageClass,
    pub(crate) ty: Box<SpirvTypeOwned>,
    pub(crate) type_reference: u32,
}
impl SpirvType for SpirvPointerType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }
    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Pointer(self)
    }
    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Pointer(self)
    }
    fn memory_size_bits(&self) -> usize {
        self.ty.memory_size_bits()
    }
    fn memory_alignment(&self) -> usize {
        self.ty.memory_alignment()
    }
}
impl SpirvUniformCollection for SpirvPointerType {
    fn len(&self) -> usize {
        1
    }
    fn element_ty(&self) -> &dyn SpirvType {
        self.ty.as_ref()
    }
    fn element_ty_mut(&mut self) -> &mut dyn SpirvType {
        self.ty.as_mut()
    }
}
impl FromInstruction for SpirvPointerType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        let id = instruction.result_id?;
        let type_id = expect_operand!(instruction.operands, 1, IdRef, "type");
        let mut inner = module
            .types
            .get(&type_id)
            .expect("pointee type missing")
            .clone();
        for (decoration, args) in module.ref_decorations(id) {
            match (&mut inner, decoration) {
                (SpirvTypeOwned::Array(array), Decoration::ArrayStride) => {
                    array.stride =
                        Some(expect_operand!(args, 0, LiteralBit32, "array stride") as usize)
                }
                (pointee, decoration) => todo!(
                    "unhandled decoration {:?} for type pointer to {:?}",
                    decoration,
                    pointee
                ),
            }
        }
        let storage_class = expect_operand!(instruction.operands, 0, StorageClass, "storage class");

        Some(SpirvPointerType {
            storage_class,
            ty: Box::new(inner),
            type_reference: id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvImageType {
    pub(crate) sampled_type: Box<SpirvTypeOwned>,
    pub(crate) dim: Dim,
    pub(crate) depth: u32,
    pub(crate) arrayed: bool,
    pub(crate) multisampled: bool,
    pub(crate) sampled: ImageSampling,
    pub(crate) format: ImageFormat,
    pub(crate) type_reference: u32,
}
impl SpirvType for SpirvImageType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }

    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::Image(self)
    }

    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::Image(self)
    }
}
impl FromInstruction for SpirvImageType {
    fn from_instruction(
        module: &ModuleExt,
        instruction: &Instruction,
        _: Option<&()>,
    ) -> Option<Self> {
        let id = instruction.result_id?;

        let sampled_type = expect_operand!(instruction.operands, 0, IdRef, "sampled type");
        let sampled_type = module.ref_type(sampled_type).expect("missing image sampled type");

        let dim = expect_operand!(instruction.operands, 1, Dim, "dimensionality");
        let depth = expect_operand!(instruction.operands, 2, LiteralBit32, "depth");
        let arrayed = expect_operand!(instruction.operands, 3, LiteralBit32, "arrayed") == 1;
        let multisampled = expect_operand!(instruction.operands, 4, LiteralBit32, "multisampled") == 1;
        let sampled = ImageSampling::from_literal(expect_operand!(instruction.operands, 5, LiteralBit32, "sampled"));
        let format = expect_operand!(instruction.operands, 6, ImageFormat, "format");

        if dim == Dim::DimSubpassData {
            assert!(
                sampled == ImageSampling::Storage &&
                format == ImageFormat::Unknown,
                "If Dim is SubpassData, Sampled must be 2, Image Format must be Unknown, and the Execution Model must be Fragment."
            )
        }

        Some(SpirvImageType {
            sampled_type: Box::new(sampled_type.clone()),
            dim,
            depth,
            arrayed,
            multisampled,
            sampled,
            format,
            type_reference: id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpirvSampledImageType {
    pub(crate) image: SpirvImageType,
    pub(crate) type_reference: u32,
}
impl SpirvType for SpirvSampledImageType {
    fn type_reference(&self) -> u32 {
        self.type_reference
    }

    fn as_typed_ref(&self) -> SpirvTypeRef<'_> {
        SpirvTypeRef::SampledImage(self)
    }

    fn as_typed_ref_mut(&mut self) -> SpirvTypeRefMut<'_> {
        SpirvTypeRefMut::SampledImage(self)
    }
}
impl FromInstruction for SpirvSampledImageType {
    fn from_instruction(
            module: &ModuleExt,
            instruction: &Instruction,
            _: Option<&()>,
        ) -> Option<Self> {
        let id = instruction.result_id?;

        let image = expect_operand!(instruction.operands, 0, IdRef, "image type");
        let image = module.ref_type(image).expect("missing image type");
        let image = match image {
            SpirvTypeOwned::Image(it) => it.clone(),
            _ => panic!("sampled image expects Image argument")
        };

        Some(SpirvSampledImageType {
            image,
            type_reference: id,
        })
    }
}
