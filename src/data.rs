use serde::{Serialize, Deserialize};
use spirv::*;
pub use spirv::StorageClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[non_exhaustive]
pub enum ShaderStage {
    Vertex = 0u32,
    TessellationControl = 1u32,
    TessellationEvaluation = 2u32,
    Geometry = 3u32,
    Fragment = 4u32,
    Compute = 5u32,
    Task = 5267u32,
    Mesh = 5268u32,
    RayGeneration = 5313u32,
    Intersection = 5314u32,
    AnyHit = 5315u32,
    ClosestHit = 5316u32,
    Miss = 5317u32,
    Callable = 5318u32,
}
impl ShaderStage {
    pub const GRAPHICS: &'static [ShaderStage] = &[
        Self::Vertex,
        Self::TessellationControl,
        Self::TessellationEvaluation,
        Self::Geometry,
        Self::Fragment,
    ];
    pub const COMPUTE: &'static [ShaderStage] = &[Self::Compute];
    pub const RAY_TRACING: &'static [ShaderStage] = &[
        Self::RayGeneration,
        Self::Intersection,
        Self::AnyHit,
        Self::ClosestHit,
        Self::Miss,
        Self::Callable,
    ];

    pub fn as_shader_stage_flag(&self) -> Option<u32> {
        Some(match self {
            Self::Vertex => 0b1,
            Self::TessellationControl => 0b10,
            Self::TessellationEvaluation => 0b100,
            Self::Geometry => 0b1000,
            Self::Fragment => 0b1_0000,
            Self::Compute => 0b10_0000,
            _ => return None,
        })
    }

    pub fn is_graphics(&self) -> bool {
        Self::GRAPHICS.contains(self)
    }
    pub fn is_compute(&self) -> bool {
        Self::COMPUTE.contains(self)
    }
    pub fn is_ray_tracing(&self) -> bool {
        Self::RAY_TRACING.contains(self)
    }
}
impl From<ExecutionModel> for ShaderStage {
    fn from(value: ExecutionModel) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}
impl From<ShaderStage> for ExecutionModel {
    fn from(val: ShaderStage) -> Self {
        unsafe { std::mem::transmute(val) }
    }
}
impl std::fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ShaderStage::Vertex => "Vertex",
            ShaderStage::TessellationControl => "TessellationControl",
            ShaderStage::TessellationEvaluation => "TessellationEvaluation",
            ShaderStage::Geometry => "Geometry",
            ShaderStage::Fragment => "Fragment",
            ShaderStage::Compute => "Compute",
            ShaderStage::Task => "Task",
            ShaderStage::Mesh => "Mesh",
            ShaderStage::RayGeneration => "RayGeneration",
            ShaderStage::Intersection => "Intersection",
            ShaderStage::AnyHit => "AnyHit",
            ShaderStage::ClosestHit => "ClosestHit",
            ShaderStage::Miss => "Miss",
            ShaderStage::Callable => "Callable",
        })
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
#[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorType.html>"]
pub enum DescriptorType {
    Sampler = 0,
    CombinedImageSampler = 1,
    SampledImage = 2,
    StorageImage = 3,
    UniformTexelBuffer = 4,
    StorageTexelBuffer = 5,
    UniformBuffer = 6,
    StorageBuffer = 7,
    UniformBufferDynamic = 8,
    StorageBufferDynamic = 9,
    InputAttachment = 10,
    /// Provided by VK_VERSION_1_3 & VK_EXT_inline_uniform_block
    InlineUniformBlock = 1000138000,
    /// Provided by VK_KHR_acceleration_structure
    AccelerationStructureKhr = 1000150000,
    /// Provided by VK_NV_ray_tracing
    AccelerationStructureNv = 1000165000,
    /// Provided by VK_QCOM_image_processing
    SampleWeightImageQcom = 1000440000,
    /// Provided by VK_QCOM_image_processing
    BlockMatchImageQcom = 1000440001,
    /// Provided by VK_EXT_mutable_descriptor_type & VK_VALVE_mutable_descriptor_type
    MutableExt = 1000351000,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[doc = "<https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage>"]
pub enum ImageSampling {
    /// indicates this is only known at run time, not at compile time
    Dynamic = 0,
    /// indicates an image compatible with sampling operations
    Sampled = 1,
    /// indicates an image compatible with read/write operations (a storage or subpass data image).
    Storage = 2,
}
impl ImageSampling {
    pub fn from_literal(value: u32) -> Self {
        match value {
            0 => Self::Dynamic,
            1 => Self::Sampled,
            2 => Self::Storage,
            other => todo!("unknown image sampling: {}", other),
        }
    }
}
