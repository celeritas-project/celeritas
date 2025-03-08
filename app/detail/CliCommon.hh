//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/CliCommon.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <CLI/CLI.hpp>

#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/io/OutputRegistry.hh"

namespace celeritas
{
namespace app
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct a failure message for celeritas apps
std::string failure_message(CLI::App const* cli, const CLI::Error& e)
{
    std::ostringstream os;
    os << cli->get_name() << ": " << CLI::FailureMessage::simple(cli, e);
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
//
char const* failure_type(std::exception const& e)
{
    if (dynamic_cast<std::runtime_error const*>(&e))
    {
        return "Runtime error";
    }
    else if (dynamic_cast<std::logic_error const*>(&e))
    {
        return "Assertion failure";
    }
    return "Unknown exception";
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// APP SETUP
//---------------------------------------------------------------------------//
//! Set up common options
inline void setup_app(CLI::App& cli)
{
    cli.failure_message(detail::failure_message);
    cli.set_version_flag("--version,-v", celeritas::version_string);
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
        .description("'-' for stdin/stdout");
}

//---------------------------------------------------------------------------//
//! Validator for the empty string
inline CLI::Validator empty_string_validator()
{
    return CLI::Validator(
               [](std::string input) {
                   return input.empty() ? std::string()
                                        : std::string("Value must be empty");
               },
               "<empty>")
        .description("Empty string");
}

//---------------------------------------------------------------------------//
// EXECUTION
//---------------------------------------------------------------------------//
//! Run, checking for errors and printing on failure
template<typename RunFunc, typename... Args>
[[nodiscard]] int run_safely(CLI::App const& cli, RunFunc&& run, Args&&... args)
{
    try
    {
        std::forward<RunFunc>(run)(std::forward<Args>(args)...);
    }
    catch (std::exception const& e)
    {
        world_logger()({cli.get_name(), 0}, LogLevel::critical)
            << failure_type(e) << ": " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------//
//! Run, printing output with exceptions if available
template<typename RunFunc, typename... Args>
[[nodiscard]] int
run_safely_with_output(CLI::App const& cli, RunFunc&& run, Args&&... args)
{
    std::shared_ptr<celeritas::OutputRegistry> output;
    int result = EXIT_FAILURE;

    try
    {
        std::forward<RunFunc>(run)(output, std::forward<Args>(args)...);
        result = EXIT_SUCCESS;
    }
    catch (std::exception const& e)
    {
        world_logger()({cli.get_name(), 0}, LogLevel::critical)
            << failure_type(e) << ": " << e.what();

        if (!output)
        {
            output = std::make_shared<OutputRegistry>();
        }
        output->insert(
            std::make_shared<ExceptionOutput>(std::current_exception()));
    }

    if (!output)
    {
        CELER_LOG(warning) << "No output available";
        std::cout << "null\n";
        return EXIT_FAILURE;
    }

    CELER_LOG(status) << "Saving output";
    output->output(&std::cout);
    std::cout << std::endl;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace app
}  // namespace celeritas
