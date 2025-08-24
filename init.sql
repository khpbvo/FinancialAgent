CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION find_similar_transactions(
    query_embedding vector(3072),
    match_count int DEFAULT 10
)
RETURNS TABLE(
    id int,
    amount float,
    description text,
    category varchar,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.amount,
        t.description,
        t.category,
        1 - (t.embedding <=> query_embedding) as similarity
    FROM transactions t
    WHERE t.embedding IS NOT NULL
    ORDER BY t.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
