# Cursor CLI Code Review Workflow

This GitHub Actions workflow uses the official [Cursor CLI](https://cursor.com/docs/cli/cookbook/code-review) to perform AI-powered code reviews on Pull Requests.

## Features

- 🤖 Uses Cursor Agent for intelligent code analysis
- 🔒 Security vulnerability detection (SQL injection, XSS, etc.)
- 🐛 Bug and logic error identification
- ⚡ Performance anti-pattern detection
- 📝 Inline comments on exact changed lines
- 🔄 Automatic resolution tracking for fixed issues
- 🚫 Optional blocking mode for critical issues

## Setup Instructions

### 1. Get a Cursor API Key

1. Visit [cursor.com](https://cursor.com) and sign in
2. Navigate to your account settings
3. Generate an API key for CLI usage

### 2. Add Repository Secret

Navigate to your repository's **Settings → Secrets and variables → Actions** and add:

| Secret Name | Required | Description |
|-------------|----------|-------------|
| `CURSOR_API_KEY` | Yes | Your Cursor API key |

> **Note**: `GITHUB_TOKEN` is automatically provided by GitHub Actions.

### 3. Configure Blocking Reviews (Optional)

To block PRs with critical issues from being merged:

1. Go to **Settings → Secrets and variables → Actions → Variables**
2. Create a new variable named `BLOCKING_REVIEW` with value `true`

## File Structure

```
.
├── .cursor/
│   └── cli.json              # Agent permissions configuration
├── .github/
│   ├── README.md             # This file
│   └── workflows/
│       └── ai-pr-review.yml  # The workflow file
```

## How It Works

### Workflow Trigger

The workflow runs automatically on:
- New pull requests
- Updates to existing pull requests (new commits)
- Reopened pull requests
- PRs marked as ready for review

**Skipped for:**
- Draft pull requests

### Review Process

1. **Checkout**: Fetches the PR code
2. **Install Cursor CLI**: Downloads and configures the Cursor CLI
3. **Configure Git**: Sets up git identity for the agent
4. **Perform Review**: Cursor Agent analyzes the diff and posts inline comments
5. **Blocking Check** (optional): Fails the workflow if critical issues are found

### What Gets Reviewed

The agent analyzes for high-severity issues only:

- ❌ Null/undefined dereferences
- ❌ Resource leaks (unclosed files/connections)
- ❌ Injection vulnerabilities (SQL/XSS)
- ❌ Concurrency/race conditions
- ❌ Missing error handling for critical operations
- ❌ Obvious logic errors
- ❌ Clear performance anti-patterns
- ❌ Security vulnerabilities

### Comment Format

Comments use emojis for quick scanning:

| Emoji | Meaning |
|-------|---------|
| 🚨 | Critical issue |
| 🔒 | Security concern |
| ⚡ | Performance issue |
| ⚠️ | Logic problem |
| ✅ | Resolved issue |
| ✨ | Improvement suggestion |

## Agent Permissions

The `.cursor/cli.json` file restricts what the agent can do:

```json
{
  "permissions": {
    "deny": [
      "Shell(git push)",
      "Shell(gh pr create)",
      "Write(**)"
    ]
  }
}
```

This ensures the agent can only:
- ✅ Read files and code
- ✅ Use GitHub CLI for comments
- ❌ Cannot push code
- ❌ Cannot create PRs
- ❌ Cannot modify files

## Configuration Options

### Change the AI Model

Edit the `MODEL` environment variable in the workflow:

```yaml
env:
  MODEL: claude-sonnet-4-20250514  # Default
  # MODEL: gpt-4o                  # Alternative
```

### Adjust Review Behavior

Modify the prompt in the workflow to:
- Add project-specific coding standards
- Focus on different issue types
- Change the maximum number of comments
- Customize emoji usage

## Troubleshooting

### Review not appearing

1. Check the Actions tab for workflow run status
2. Verify `CURSOR_API_KEY` is correctly set
3. Ensure the PR is not a draft

### Agent not commenting

1. Check workflow logs for errors
2. Verify the agent has proper permissions
3. Ensure the diff is not empty

### Blocking review not working

1. Verify `BLOCKING_REVIEW` variable is set to `true`
2. Check that critical issues (🚨 or 🔒) were actually found

## References

- [Cursor CLI Documentation](https://cursor.com/docs/cli/cookbook/code-review)
- [Cursor CLI Permissions Reference](https://cursor.com/docs/cli/reference/permissions)
- [Bugbot - Managed Alternative](https://cursor.com/docs/bugbot)

## Alternative: Bugbot

For a zero-configuration managed solution, consider using [Bugbot](https://cursor.com/docs/bugbot) instead. Bugbot provides automated code review with no setup required.
