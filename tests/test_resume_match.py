from src.retrieval import resume_match


# ---------------------------
# Mock JobSearchService
# ---------------------------
class MockJobSearchService:
    def search_jobs(self, query, top_k=5):
        return [
            {
                "job_id": "1",
                "title": "ML Engineer",
                "company": "TestCorp",
                "location": "Remote",
                "url": "http://example.com/job1",
                "score": 0.95,
            }
        ]


# ---------------------------
# Mock fetch_job_detail
# ---------------------------
def mock_fetch_job_detail(job_id):
    return {
        "skills": ["Python", "Docker", "AWS"]
    }


# ---------------------------
# Test
# ---------------------------
def test_match_resume_to_jobs(monkeypatch):
    # JobSearchService
    monkeypatch.setattr(
        resume_match,
        "JobSearchService",
        lambda: MockJobSearchService()
    )

    # fetch_job_detail
    monkeypatch.setattr(
        resume_match,
        "fetch_job_detail",
        mock_fetch_job_detail
    )

    resume_text = "I have experience with Python, Docker, and AWS."

    result = resume_match.match_resume_to_jobs(resume_text, top_k=1)

    # -------- Basic --------
    assert result is not None
    assert "resume_profile" in result
    assert "matches" in result

    # -------- resume profile --------
    assert "skills" in result["resume_profile"]
    assert "python" in result["resume_profile"]["skills"]

    # -------- matches --------
    matches = result["matches"]
    assert isinstance(matches, list)
    assert len(matches) == 1

    match = matches[0]

    assert match["job_id"] == "1"
    assert match["semantic_score"] > 0

    # -------- Core Fields --------
    assert "shared_skills" in match
    assert "missing_skills" in match
    assert "match_reasons" in match

    # -------- Logical Validation --------
    assert "python" in match["shared_skills"]