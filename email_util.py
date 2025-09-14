import email
import imaplib
import os
from email.policy import default

# LangChain loaders for attachments
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

from cleaner import clean_html

# ---------- CONFIG ----------
EMAIL_HOST = "imap.gmail.com"
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
# ----------------------------


def fetch_emails(limit=10, since_uid=1):

    """Fetch latest emails using IMAP"""
    conn = imaplib.IMAP4_SSL(EMAIL_HOST)
    conn.login(EMAIL_USER, EMAIL_PASS)
    conn.select("inbox")

    # Fetch all sequence numbers in the mailbox
    result, data = conn.search(None, "ALL")
    seq_nums = [int(s) for s in data[0].split()]

    # Only process sequence numbers after last_seq
    seq_nums_to_process = [s for s in seq_nums if s > since_uid]

    # Apply limit
    seq_nums_to_process = seq_nums_to_process[:limit]

    emails = []
    for seq in seq_nums_to_process:
        # Fetch full email by sequence number
        _, msg_data = conn.fetch(str(seq), "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1], policy=default)

        # Fetch UID for deduplication/metadata
        _, uid_data = conn.fetch(str(seq), "(UID)")
        emails.append((seq, msg))

    conn.close()
    conn.logout()
    return emails


def extract_email_content(msg):
    """Extract plain text and attachments, cleaning up temp files."""
    texts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                texts.append(part.get_payload(decode=True).decode(errors="ignore"))
            if ctype == "text/html":
                payload = part.get_payload(decode=True).decode(errors="ignore")
                texts.append(clean_html(payload))
            elif part.get_filename():
                # Handle attachment
                fname = part.get_filename()
                data = part.get_payload(decode=True)
                temp_path = f"/tmp/{fname}"

                try:
                    with open(temp_path, "wb") as f:
                        f.write(data)

                    # Load content based on file type
                    if fname.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    elif fname.endswith(".docx"):
                        loader = UnstructuredWordDocumentLoader(temp_path)
                    elif fname.endswith(".txt"):
                        loader = TextLoader(temp_path)
                    else:
                        continue  # skip unsupported

                    docs = loader.load()
                    texts.extend([d.page_content for d in docs])
                finally:
                    # Remove the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True).decode(errors="ignore")

        if ctype == "text/html":
            payload = clean_html(payload)  # strip tags with BeautifulSoup
        texts.append(payload)

    return "\n".join(texts)

def create_conn():
    """Fetch latest emails using IMAP"""
    conn = imaplib.IMAP4_SSL(EMAIL_HOST)
    conn.login(EMAIL_USER, EMAIL_PASS)
    conn.select("inbox")
    return conn


