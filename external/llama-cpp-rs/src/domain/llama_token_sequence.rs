use crate::domain::LTokenSequence;
use crate::LToken;
use std::fmt::{Debug, Formatter};
use std::ptr;

impl Default for LTokenSequence {
    fn default() -> Self {
        LTokenSequence::new()
    }
}

impl LTokenSequence {
    pub fn new() -> LTokenSequence {
        LTokenSequence { tokens: Vec::new() }
    }

    /// Increase the manifest allocation of tokens in this sequence to length.
    pub fn resize(&mut self, length: usize) {
        if self.tokens.len() > length {
            self.tokens.truncate(length);
        } else {
            let missing_capacity = length - self.tokens.len();
            for _ in 0..missing_capacity {
                self.tokens.push(LToken::default_token());
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn capacity(&self) -> usize {
        self.tokens.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }



    pub fn push(&mut self, token: LToken) {
        let value = unsafe { token.native_value() };
        self.tokens.push(value);
    }

    pub fn clear(&mut self) {
        if self.is_empty() {
            return;
        }
        unsafe {
            ptr::write_bytes(self.tokens.as_mut_ptr(), 0, self.len());
        }
    }

    pub(crate) fn slice(&self, start_idx: usize) -> Self {
        Self {
            tokens: self.tokens.as_slice()[start_idx..].to_vec(),
        }
    }

    pub(crate) unsafe fn native_ptr(&self) -> *const llama_cpp_sys::llama_token {
        self.tokens.as_ptr()
    }

    pub(crate) unsafe fn native_mut_ptr(&mut self) -> *mut llama_cpp_sys::llama_token {
        self.tokens.as_mut_ptr()
    }

    pub(crate) unsafe fn native_ptr_offset(
        &self,
        offset: usize,
    ) -> *const llama_cpp_sys::llama_token {
        self.tokens.as_ptr().add(offset)
    }
}

impl Debug for LTokenSequence {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let id_stream: Vec<isize> = self.tokens.iter().map(|f| *f as isize).collect();
        write!(f, "{:?}", id_stream)
    }
}
