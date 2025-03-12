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
        return "runtime error";
    }
    else if (dynamic_cast<std::logic_error const*>(&e))
    {
        return "assertion failure";
    }
    return "unknown exception";
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Process a parsing error.
 */
[[nodiscard]] int process_parse_error(CLI::ParseError const& e)
{
    if (e.get_exit_code() != EXIT_SUCCESS)
    {
        world_logger()({cli_app().get_name(), 0}, LogLevel::critical)
            << e.get_name() << ": " << e.what();
        if (celeritas::comm_world().rank() == 0)
        {
            print_usage(cli_app(), std::clog);
        }
    }
    else if (celeritas::comm_world().rank() == 0)
    {
        return cli_app().exit(e);
    }
    return e.get_exit_code();
}

//---------------------------------------------------------------------------//
/*!
 * Process a runtime error from run_safely.
 */
[[nodiscard]] int process_runtime_error(std::exception const& e)
{
    auto msg = world_logger()({cli_app().get_name(), 0}, LogLevel::critical);

    if (!dynamic_cast<RuntimeError const*>(&e))
    {
        // Not a Celeritas runtime error: print exception type
        msg << failure_type(e) << ": ";
    }
    msg << e.what();

    return EXIT_FAILURE;
}

//---------------------------------------------------------------------------//
/*!
 * Access the app, setting common options.
 */
CLI::App& cli_app()
{
    static CLI::App result;
    static bool const did_setup = [&] {
        result.failure_message(failure_message);
        result.set_version_flag("--version,-v", celeritas::version_string);
        return true;
    }();
    CELER_ENSURE(did_setup);
    return result;
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
}  // namespace app
}  // namespace celeritas
