//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/CliCommon.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLI/CLI.hpp>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace app
{
namespace detail
{

//---------------------------------------------------------------------------//
//! Run, checking for errors and printing on failure
template<typename RunFunc, typename... Args>
[[nodiscard]] int run_safely(RunFunc&& run, Args&&... args)
{
    try
    {
        std::forward<RunFunc>(run)(std::forward<Args>(args)...);
    }
    catch (RuntimeError const& e)
    {
        CELER_LOG(critical) << "Runtime error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (DebugError const& e)
    {
        CELER_LOG(critical) << "Assertion failure: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------//

//! Construct a failure message for celeritas apps
inline std::string failure_message(const CLI::App* cli, const CLI::Error& e)
{
    std::ostringstream os;
    os << cli->get_name() << ": " << CLI::FailureMessage::simple(cli, e);
    return std::move(os).str();
}

//---------------------------------------------------------------------------//

//! Validator for the special value '-'
inline CLI::Validator dash_validator()
{
    return CLI::Validator(
               [](std::string input) {
                   return input == "-" ? std::string()
                                       : std::string("Value must be '-'");
               },
               "-")
        .description("Special value '-' for stdin/stdout");
}

//! Validator for the empty string
inline CLI::Validator empty_string_validator()
{
    return CLI::Validator(
               [](std::string input) {
                   return input.empty() ? std::string()
                                        : std::string("Value must be empty");
               },
               "EMPTY")
        .description("Empty string");
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace app
}  // namespace celeritas
