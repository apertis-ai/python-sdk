## ADDED Requirements

### Requirement: Text to speech
`client.audio.speech.create` SHALL post `model`, `input`, `voice` and any given optional
fields as JSON to `/audio/speech` on sync and async clients, and return an object
exposing the raw audio bytes and a `write_to_file(path)` helper.

#### Scenario: Save speech
- **WHEN** a caller creates speech and calls `write_to_file`
- **THEN** the file holds exactly the bytes the API returned

### Requirement: Transcription and translation uploads
`client.audio.transcriptions.create` and `client.audio.translations.create` SHALL send
the audio file and fields as `multipart/form-data` to `/audio/transcriptions` and
`/audio/translations`, accepting a path, an open binary file, raw bytes, or a
`(filename, bytes)` tuple. They SHALL return a typed object with `text` for the
`json` and `verbose_json` formats, keeping any extra fields, and a `str` for the
`text`, `srt` and `vtt` formats.

#### Scenario: Upload from a path
- **WHEN** a caller passes a file path
- **THEN** the request is multipart with the file under `file` and its base name as the
  filename, and the response `text` is returned

#### Scenario: Retry resends the file
- **WHEN** the first attempt gets a retryable error
- **THEN** the retry sends the same file bytes

#### Scenario: Subtitle format
- **WHEN** `response_format="srt"`
- **THEN** the call returns the response body as a string
