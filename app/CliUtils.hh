//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CliUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>
#include <exception>
#include <type_traits>
#include <CLI/CLI.hpp>

//! Parse but only print on one processor on failure/help
#define CELER_CLI11_PARSE(ARGC, ARGV)                  \
    try                                                \
    {                                                  \
        ::celeritas::app::cli_app().parse(ARGC, ARGV); \
    }                                                  \
    catch (CLI::ParseError const& e)                   \
    {                                                  \
        return celeritas::app::process_parse_error(e); \
    }

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

// Process a parsing error
[[nodiscard]] int process_parse_error(CLI::ParseError const&);

// Process a runtime error (returning EXIT_FAILURE)
[[nodiscard]] int process_runtime_error(std::exception const& e);

// Access the app instance
CLI::App& cli_app();

// Get a validator for the special value '-'
CLI::Validator dash_validator();

// Get a validator for the empty string
CLI::Validator empty_string_validator();

//! Raise an error about
class ConflictingArguments : public CLI::ArgumentMismatch
{
  public:
    explicit ConflictingArguments(std::string const& msg)
        : CLI::ArgumentMismatch("conflicting or missing arguments",
                                msg,
                                CLI::ExitCodes::ArgumentMismatch)
    {
    }
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
//! Run, checking for errors, printing on failure and returning an exit code
template<typename RunFunc, typename... Args>
[[nodiscard]] inline int run_safely(RunFunc&& run, Args&&... args)
{
    static_assert(std::is_void_v<std::invoke_result_t<RunFunc, Args...>>,
                  "RunFunc must return void");
    try
    {
        std::forward<RunFunc>(run)(std::forward<Args>(args)...);
    }
    catch (std::exception const& e)
    {
        return process_runtime_error(e);
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
