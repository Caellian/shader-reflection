# SPIR-V Shader Reflection

A library that reads SPIR-V binaries and provides binding descriptors, variable type information,
constant value inspection, etc. at compile-time and runtime.

This streamlines experimentation with shaders and makes it simpler to keep the Vulkan pipeline
configuration correct by shifting errors from program execution to compilation and making them
much more obvious/clear.

## Features

- Provide information needed for construction of binding descriptors
- Generate types for bindings and push constants
  - `glam` integration
- [ ] Handle shader source conversion using `naga`
- [ ] Bake in optimization using [Embark Studios spirv tools wrapper](https://github.com/EmbarkStudios/spirv-tools-rs).
- [ ] Provide a macro for embedding metadata and SPIR-V bytecode into target binaries

## Attribution

- Binding descriptor inference code is based on [`rspirv-reflect`](https://github.com/Traverse-Research/rspirv-reflect).
- SPIR-V binary normalization utility function is copied from [`ash`](https://github.com/ash-rs/ash).

## License

This library is licensed under terenary MIT, Apache 2.0, Zlib license.
Copies of the licenses are provided in the root of repository.
