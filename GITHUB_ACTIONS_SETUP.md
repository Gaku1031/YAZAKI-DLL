# GitHub Actions CI/CD Setup Guide

## Overview

This repository includes a complete CI/CD pipeline that:
- ✅ Builds the Blood Pressure DLL on Windows
- ✅ Runs C# integration tests
- ✅ Prevents merging if tests fail
- ✅ Provides downloadable DLL artifacts

## Files Created

### Workflow Configuration
- `.github/workflows/build-and-test.yml` - Main CI/CD pipeline
- `.github/PULL_REQUEST_TEMPLATE.md` - PR template with checklist

### Build Scripts
- `scripts/build-dll.ps1` - PowerShell build script for Windows
- `tests/CSharpTest/BloodPressureTest.csproj` - C# test project
- `tests/CSharpTest/Program.cs` - Integration test suite

### Test Data
- `tests/test-data/` - Directory for test video files
- `tests/test-data/README.md` - Test data documentation

## Setting Up PR Protection Rules

### 1. Enable Branch Protection

Go to your repository settings:
1. Navigate to **Settings** → **Branches**
2. Click **Add rule** for your main branch
3. Configure the following:

#### Required Settings
```
Branch name pattern: main
☑️ Require a pull request before merging
☑️ Require status checks to pass before merging
☑️ Require branches to be up to date before merging
☑️ Require conversation resolution before merging
☑️ Include administrators
```

#### Required Status Checks
Add these status checks:
- `build-and-test` (from the workflow)

### 2. Workflow Protection

The workflow will automatically:
- ❌ **FAIL** if DLL build fails
- ❌ **FAIL** if C# compilation fails  
- ❌ **FAIL** if integration tests fail
- ✅ **PASS** only if all tests succeed

### 3. Manual Setup Steps

#### Repository Secrets (if needed)
Currently no secrets are required, but you may add:
- `GITHUB_TOKEN` (automatically provided)

#### Required Repository Settings
1. **Actions permissions**: Enable GitHub Actions
2. **Artifact retention**: Set to 30 days (configurable)

## Workflow Details

### Trigger Events
```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```

### Build Process
1. **Setup Environment**
   - Windows latest runner
   - Python 3.11 x64
   - Visual Studio build tools
   - .NET 6.0

2. **Build DLL**
   - Install Python dependencies
   - Generate C++ wrapper code
   - Compile DLL with MSVC
   - Verify exports with dumpbin

3. **Test Execution**
   - Build C# test project
   - Copy DLL and dependencies
   - Run integration tests
   - Validate test results

4. **Artifact Generation**
   - Package DLL with dependencies
   - Upload as downloadable artifacts
   - Include README with build info

### Artifacts Available

#### Distribution Package (`BloodPressureEstimation-DLL-{SHA}`)
Contains:
- `BloodPressureEstimation.dll` - Main DLL
- `bp_estimation_simple.py` - Python module
- `python311.dll` - Python runtime
- `vcruntime140*.dll` - C++ runtime
- `README.txt` - Usage instructions

#### Test Results (`test-results-{SHA}`)
Contains:
- Test execution logs
- Any generated CSV files
- Debug information

#### Build Logs (`build-logs-{SHA}`) - Only on failure
Contains:
- Compilation logs
- Error details
- Build artifacts

## Using the Artifacts

### Downloading from GitHub
1. Go to **Actions** tab in your repository
2. Click on a successful workflow run
3. Scroll down to **Artifacts** section
4. Download `BloodPressureEstimation-DLL-{SHA}`

### Integration in C# Projects
```csharp
// 1. Extract downloaded artifacts to your C# project directory
// 2. Add DLL reference or use P/Invoke as shown in tests
// 3. Ensure all dependency DLLs are in the same directory

[DllImport("BloodPressureEstimation.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern bool InitializeDLL(string modelDir);
```

## Test Coverage

### Integration Tests Include
- ✅ Environment validation
- ✅ DLL loading verification
- ✅ Function export validation
- ✅ Version information retrieval
- ✅ Status checking functionality
- ✅ Basic blood pressure analysis
- ✅ Error handling verification

### Test Data Strategy
- Uses real video files if available in `tests/test-data/`
- Falls back to dummy data for basic functionality testing
- Tests are designed to be resilient to missing real data

## Troubleshooting

### Common Issues

#### 1. Build Failures
**Symptom**: DLL compilation fails
**Solutions**:
- Check Python dependencies in `requirements_balanced_20mb.txt`
- Verify C++ source files are generated correctly
- Check Visual Studio environment setup

#### 2. Test Failures
**Symptom**: C# tests fail to run
**Solutions**:
- Verify DLL exports with `dumpbin /exports`
- Check if all dependency DLLs are copied
- Review test logs for specific error messages

#### 3. Missing Artifacts
**Symptom**: No artifacts uploaded
**Solutions**:
- Check if workflow completed successfully
- Verify artifact upload steps didn't fail
- Check repository artifact retention settings

### Manual Local Testing

To test locally before pushing:
```powershell
# 1. Run the build script
.\scripts\build-dll.ps1

# 2. Build the test project
dotnet build tests/CSharpTest/BloodPressureTest.csproj

# 3. Copy files and run tests
# (Follow the workflow steps manually)
```

## Monitoring

### Workflow Status
- Check the **Actions** tab for build status
- Green checkmark = All tests passed
- Red X = Build or tests failed
- Yellow circle = In progress

### Notifications
Configure notifications in your GitHub settings:
- **Settings** → **Notifications** → **Actions**
- Enable notifications for workflow failures

## Security Considerations

### Code Safety
- All builds run in isolated GitHub runners
- No secrets or sensitive data are exposed
- Artifacts are publicly downloadable (be aware of this)

### Dependency Management
- Python dependencies are pinned to specific versions
- Visual C++ runtime is included for portability
- No network calls during DLL operation (except for Python imports)

## Maintenance

### Regular Updates
- Monitor for security updates to dependencies
- Update Python version as needed
- Keep GitHub Actions versions current

### Performance Optimization
- Build cache is used for Python dependencies
- Artifacts have reasonable retention periods
- Parallel execution where possible

This setup ensures that your Blood Pressure DLL is always buildable, testable, and deployable through GitHub Actions!