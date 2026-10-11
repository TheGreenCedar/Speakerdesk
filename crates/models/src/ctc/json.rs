//! JSON subset parser for the alignment manifest. No extra crate dependency.
use super::manifest::ManifestError;

#[derive(Debug)]
pub(super) enum Json {
    Null,
    Bool,
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

pub(super) fn parse(input: &str) -> Result<Json, ManifestError> {
    let mut parser = Parser { input, pos: 0 };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.input.len() {
        return Err(parser.fail("trailing data"));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn fail(&self, message: &str) -> ManifestError {
        ManifestError::new(format!("{message} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.bump();
        }
    }

    fn parse_value(&mut self) -> Result<Json, ManifestError> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Json::String(self.parse_string()?)),
            Some('-' | '0'..='9') => Ok(Json::Number(self.parse_number()?)),
            Some('t') => self.parse_literal("true"),
            Some('f') => self.parse_literal("false"),
            Some('n') => self.parse_literal("null"),
            _ => Err(self.fail("invalid JSON value")),
        }
    }

    fn parse_literal(&mut self, literal: &str) -> Result<Json, ManifestError> {
        if !self.input[self.pos..].starts_with(literal) {
            return Err(self.fail("invalid JSON literal"));
        }
        self.pos += literal.len();
        match literal {
            "true" | "false" => Ok(Json::Bool),
            "null" => Ok(Json::Null),
            _ => Err(self.fail("invalid JSON literal")),
        }
    }

    fn parse_object(&mut self) -> Result<Json, ManifestError> {
        self.bump();
        let mut fields = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some('}') {
                self.bump();
                return Ok(Json::Object(fields));
            }
            if !fields.is_empty() {
                if self.peek() != Some(',') {
                    return Err(self.fail("expected comma"));
                }
                self.bump();
                self.skip_ws();
            }
            if self.peek() != Some('"') {
                return Err(self.fail("expected object key"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump() != Some(':') {
                return Err(self.fail("expected colon"));
            }
            fields.push((key, self.parse_value()?));
        }
    }

    fn parse_array(&mut self) -> Result<Json, ManifestError> {
        self.bump();
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                self.bump();
                return Ok(Json::Array(items));
            }
            if !items.is_empty() {
                if self.peek() != Some(',') {
                    return Err(self.fail("expected comma"));
                }
                self.bump();
                self.skip_ws();
            }
            items.push(self.parse_value()?);
        }
    }

    fn parse_string(&mut self) -> Result<String, ManifestError> {
        if self.bump() != Some('"') {
            return Err(self.fail("expected string"));
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                Some('"') => return Ok(out),
                Some('\\') => out.push(self.parse_escape()?),
                Some(ch) if (ch as u32) < 0x20 => return Err(self.fail("raw control in string")),
                Some(ch) => out.push(ch),
                None => return Err(self.fail("unterminated string")),
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, ManifestError> {
        match self.bump() {
            Some('"') => Ok('"'),
            Some('\\') => Ok('\\'),
            Some('/') => Ok('/'),
            Some('b') => Ok('\u{0008}'),
            Some('f') => Ok('\u{000c}'),
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('u') => {
                let start = self.pos;
                for _ in 0..4 {
                    match self.bump() {
                        Some(ch) if ch.is_ascii_hexdigit() => {}
                        _ => return Err(self.fail("invalid unicode escape")),
                    }
                }
                let code = u32::from_str_radix(&self.input[start..self.pos], 16)
                    .map_err(|_| self.fail("invalid unicode escape"))?;
                char::from_u32(code).ok_or_else(|| self.fail("invalid unicode escape"))
            }
            _ => Err(self.fail("invalid escape")),
        }
    }

    fn parse_number(&mut self) -> Result<String, ManifestError> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.bump();
        }
        match self.bump() {
            Some('0') => {}
            Some(ch) if ch.is_ascii_digit() => {
                while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                    self.bump();
                }
            }
            _ => return Err(self.fail("invalid number")),
        }
        if self.peek() == Some('.') {
            self.bump();
            let mut digits = 0;
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump();
                digits += 1;
            }
            if digits == 0 {
                return Err(self.fail("invalid number"));
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            let mut digits = 0;
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump();
                digits += 1;
            }
            if digits == 0 {
                return Err(self.fail("invalid number"));
            }
        }
        Ok(self.input[start..self.pos].to_string())
    }
}
