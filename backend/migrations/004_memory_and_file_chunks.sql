ALTER TABLE messages ADD COLUMN memory_used_json TEXT NOT NULL DEFAULT '[]';

CREATE TABLE upload_chunks (
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL CHECK (chunk_number >= 0),
    content TEXT NOT NULL,
    PRIMARY KEY (upload_id, chunk_number)
);
CREATE INDEX upload_chunks_upload_id_idx ON upload_chunks(upload_id, chunk_number);
