use std::path::{PathBuf, Path};

use super::{Error, SpirVModule};

pub trait ShaderSource {
    fn compile(&self) -> Result<SpirVModule, Error>;
}

pub struct WGSLShader {
    source_file: PathBuf,
    source: String,
}

impl WGSLShader {
    pub fn read(path: impl AsRef<Path>) -> Result<WGSLShader, Error> {
        let source_file = path.as_ref().to_path_buf();

        let source = std::fs::read_to_string(&source_file)?;

        Ok(Self {
            source_file,
            source
        })
    }
}

impl ShaderSource for WGSLShader {
    fn compile(&self) -> Result<SpirVModule, Error> {
        todo!()
    }
}
