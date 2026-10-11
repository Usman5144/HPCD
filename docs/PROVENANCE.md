# Frozen-source provenance and path redaction

The scientific freeze artifact `configs/hpcd/FINAL_METHOD_FREEZE.json` records the original pre-release SHA-256 checksums and the experimental settings. These checksums apply to the **original** author-machine bytes, not to the released files after required identity/path scrubbing. All four pinned code files and the HJTM artifact were verified byte-for-byte against the freeze *before* redaction. Changes to the publication copy have been confined to absolute filesystem paths and the installation prefix, not algorithmic constants.

| Frozen component | Original SHA-256 |
|---|---|
| Reference-aligned controller | `177b1d5165a0da67fdc51f8750877839ce537e170993c36d52a74ffd5f791949` |
| Development generation | `0f267fa31f4b58d2365163c94324ed8313a680d17eb3bceac19b2bc5ed9c7845` |
| Quality scoring | `9018a89c94770c584f8ec29df510be02f4ba5329bb38a6fdb97d77897e7c614f` |
| Entity selection finalizer | `a3e9a1f4a6b2043c906465185012dc1dd5e2cd7a1b39b6ccc2277a6ef70bff74` |
| HJTM model JSON | `1abd60c38a8787c4d44cf4e81ec157eaf18745ce1dec28c366417030b1a22c61` |

The release does not include original unsanitized scripts, commits, private environment paths or author credentials. No historical git history should be imported when uploading. For independent attestation, authors should retain their private originals and the accompanying frozen manifests, not expose them in the anonymized repository.
