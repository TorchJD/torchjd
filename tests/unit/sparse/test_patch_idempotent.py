from torchjd.sparse._patch import enable_seamless_sparse

def test_enable_patch_idempotent():
    enable_seamless_sparse()   # first call patches
    enable_seamless_sparse()   # second call should be a no-op
