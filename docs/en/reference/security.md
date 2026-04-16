# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of TensorCraft-HPC seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to [INSERT SECURITY EMAIL].

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, memory corruption, denial of service, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 90 days (depending on complexity)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report.

2. **Communication**: We will keep you informed of the progress towards a fix and full announcement.

3. **Credit**: We will credit you in the security advisory if you wish (please let us know your preference).

4. **Disclosure**: We follow a coordinated disclosure process. We ask that you give us reasonable time to address the issue before any public disclosure.

## Security Best Practices for Users

When using TensorCraft-HPC in your projects:

1. **Keep Updated**: Always use the latest stable version to ensure you have the latest security patches.

2. **Input Validation**: Validate all inputs before passing them to TensorCraft-HPC functions, especially when dealing with user-provided data.

3. **Memory Safety**: Be aware of memory allocation patterns and ensure proper cleanup to prevent memory leaks.

4. **CUDA Security**: Follow NVIDIA's security guidelines for CUDA applications.

## Security Updates

Security updates will be released as patch versions and announced through:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- GitHub Releases

## Preferred Languages

We prefer all communications to be in English.
