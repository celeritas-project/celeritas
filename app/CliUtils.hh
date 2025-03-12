//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CliUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <CLI/CLI.hpp>

//! Parse but only print on one processor on failure/help
#define CELER_CLI11_PARSE(CLI_APP, ...)                           \
    try                                                           \
    {                                                             \
        (CLI_APP).parse(__VA_ARGS__);                             \
    }                                                             \
    catch (CLI::ParseError const& e)                              \
    {                                                             \
        return celeritas::app::process_parse_error((CLI_APP), e); \
    }

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

// Process a parsing error
[[nodiscard]] int process_parse_error(CLI::App const&, CLI::ParseError const&);

// Process a runtime error (returning EXIT_FAILURE)
[[nodiscard]] int
process_runtime_error(CLI::App const&, std::exception const& e);

// Set up common options
void setup_app(CLI::App& cli);

// Get a validator for the special value '-'
CLI::Validator dash_validator();

// Get a validator for the empty string
CLI::Validator empty_string_validator();

// Print the usage of the app if possible, returning success
bool print_usage(CLI::App const& cli, std::ostream& os);

//! Raise an error about
class ConflictingArguments : public CLI::ArgumentMismatch
{
  public:
    explicit ConflictingArguments(std::string const& msg)
        : CLI::ArgumentMismatch(
              "conflicting arguments", msg, CLI::ExitCodes::ArgumentMismatch)
    {
    }
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
//! Run, checking for errors and printing on failure
template<typename RunFunc, typename... Args>
[[nodiscard]] inline int
run_safely(CLI::App const& cli, RunFunc&& run, Args&&... args)
{
    try
    {
        std::forward<RunFunc>(run)(std::forward<Args>(args)...);
    }
    catch (std::exception const& e)
    {
        return process_runtime_error(cli, e);
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
