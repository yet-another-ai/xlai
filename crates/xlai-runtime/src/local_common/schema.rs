use serde_json::Value;
use xlai_core::{ErrorKind, StructuredOutput, StructuredOutputFormat, XlaiError};

pub fn validate_structured_output_schema(
    structured_output: &StructuredOutput,
) -> Result<(), XlaiError> {
    match &structured_output.format {
        StructuredOutputFormat::JsonSchema { schema } => jsonschema::validator_for(schema)
            .map(|_| ())
            .map_err(|error| {
                XlaiError::new(
                    ErrorKind::Validation,
                    format!("structured output schema is invalid: {error}"),
                )
            }),
        StructuredOutputFormat::LarkGrammar { grammar } => {
            if grammar.trim().is_empty() {
                return Err(XlaiError::new(
                    ErrorKind::Validation,
                    "structured output Lark grammar must not be empty",
                ));
            }
            Ok(())
        }
    }
}

pub fn validate_structured_output(
    structured_output: &StructuredOutput,
    generated: &str,
) -> Result<(), XlaiError> {
    if let StructuredOutputFormat::JsonSchema { schema } = &structured_output.format {
        let generated = generated.trim();
        let value: Value = serde_json::from_str(generated).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("structured output was not valid JSON: {error}"),
            )
        })?;
        let validator = jsonschema::validator_for(schema).map_err(|error| {
            XlaiError::new(
                ErrorKind::Validation,
                format!("structured output schema is invalid: {error}"),
            )
        })?;
        if let Err(error) = validator.validate(&value) {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                format!("structured output did not match the requested schema: {error}"),
            ));
        }
    }
    Ok(())
}
