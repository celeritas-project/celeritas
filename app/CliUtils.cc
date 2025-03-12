//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CliUtils.cc
//---------------------------------------------------------------------------//
#include "CliUtils.hh"

#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/MpiCommunicator.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct a failure message for celeritas apps
std::string failure_message(CLI::App const* cli, const CLI::Error& e)
{
    std::ostringstream os;
    os << cli->get_name() << ": ";
    if (print_usage(*cli, os))
    {
        // Usage printed successfully; now write the error
        os << e.what();
    }
    else
    {
        // No usage available> write default error message
        os << CLI::FailureMessage::simple(cli, e);
    }

    return std::move(os).str();
}

//---------------------------------------------------------------------------//
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
/*!
 * Process a parsing error.
 */
[[nodiscard]] int
process_parse_error(CLI::App const& cli, CLI::ParseError const& e)
{
    if (e.get_exit_code() != EXIT_SUCCESS)
    {
        CELER_LOG(error) << e.get_name() << ": " << e.what();
        if (celeritas::comm_world().rank() == 0)
        {
            print_usage(cli, std::clog);
        }
    }
    else if (celeritas::comm_world().rank() == 0)
    {
        return cli.exit(e);
    }
    return e.get_exit_code();
}

//---------------------------------------------------------------------------//
/*!
 * Process a runtime error from run_safely.
 */
[[nodiscard]] int
process_runtime_error(CLI::App const& cli, std::exception const& e)
{
    world_logger()({cli.get_name(), 0}, LogLevel::critical)
        << failure_type(e) << ": " << e.what();

    return EXIT_FAILURE;
}

//---------------------------------------------------------------------------//
/*!
 * Set up common options.
 */
void setup_app(CLI::App& cli)
{
    cli.failure_message(failure_message);
    cli.set_version_flag("--version,-v", celeritas::version_string);
}

//---------------------------------------------------------------------------//
/*!
 * Get a validator for the special value '-'.
 */
CLI::Validator dash_validator()
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
/*!
 * Get a validator for the empty string.
 */
CLI::Validator empty_string_validator()
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
/*!
 * Print the usage of the app if possible, returning success.
 */
bool print_usage(CLI::App const& cli, std::ostream& os)
{
    if (auto base_formatter
        = std::dynamic_pointer_cast<CLI::Formatter>(cli.get_formatter()))
    {
        auto usage = base_formatter->make_usage(&cli, std::string{});
        if (!usage.empty() && usage.back() == '\n')
        {
            usage.pop_back();
        }
        os << usage;
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
