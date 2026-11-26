// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion_common::HashMap;
use datafusion_execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion_execution::cache::cache_manager::FileMetadata;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use object_store::ObjectMeta;
use vortex::file::Footer;
use vortex::file::VortexFile;

/// The Vortex footer `cache` holds for `object`, if the entry still matches `object`'s size and
/// last-modified time.
pub fn cached_footer(cache: &FileMetadataCache, object: &ObjectMeta) -> Option<Footer> {
    cache
        .get(&object.location)
        .filter(|entry| entry.is_valid_for(object))
        .and_then(|entry| {
            entry
                .file_metadata
                .as_any()
                .downcast_ref::<CachedVortexMetadata>()
                .map(|metadata| metadata.footer().clone())
        })
}

/// Caches the footer of `vortex_file`, opened from `object`, so [`cached_footer`] returns it.
pub fn cache_footer(cache: &FileMetadataCache, object: &ObjectMeta, vortex_file: &VortexFile) {
    cache.put(
        &object.location,
        CachedFileMetadataEntry::new(
            object.clone(),
            Arc::new(CachedVortexMetadata::new(vortex_file)),
        ),
    );
}

/// Cached Vortex file metadata for use with DataFusion's [`FileMetadataCache`].
///
/// [`FileMetadataCache`]: datafusion_execution::cache::cache_manager::FileMetadataCache
pub struct CachedVortexMetadata {
    footer: Footer,
}

impl CachedVortexMetadata {
    /// Create a new cached metadata entry from a VortexFile.
    pub fn new(vortex_file: &VortexFile) -> Self {
        let footer = vortex_file.footer();
        Self {
            footer: footer.clone(),
        }
    }

    /// Get the cached footer.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }
}

impl FileMetadata for CachedVortexMetadata {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn memory_size(&self) -> usize {
        self.footer
            .approx_byte_size()
            // 64KB is not an insane estimate...
            // We just want to avoid returning zero and _never_ being evicted from the cache.
            .unwrap_or(1024 * 64)
    }

    fn extra_info(&self) -> HashMap<String, String> {
        Default::default()
    }
}
