//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/ScopedLogStorer.cc
//---------------------------------------------------------------------------//
#include "ScopedLogStorer.hh"

#include <iostream>
#include <regex>

#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/io/Repr.hh"
#include "corecel/sys/Environment.hh"

#include "StringSimplifier.hh"

namespace celeritas
{
namespace test
{
namespace
{
void debug_clog(LogProvenance, LogLevel lev, std::string msg)
{
    std::clog << color_code('x') << to_cstring(lev) << ": " << msg
              << color_code(' ') << std::endl;
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct reference to log to temporarily replace.
 */
ScopedLogStorer::ScopedLogStorer(Logger* orig, LogLevel min_level)
    : logger_{orig}, min_level_{min_level}
{
    CELER_EXPECT(logger_);
    CELER_EXPECT(min_level != LogLevel::size_);
    // Create a new logger that calls our operator(), replace orig and store
    saved_logger_ = std::make_unique<Logger>(
        std::exchange(*logger_, Logger{std::ref(*this), LogLevel::debug}));
}

//---------------------------------------------------------------------------//
/*!
 * Construct reference to log to temporarily replace.
 */
ScopedLogStorer::ScopedLogStorer(Logger* orig)
    : ScopedLogStorer{orig, Logger::default_level()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Restore original logger on destruction.
 */
ScopedLogStorer::~ScopedLogStorer()
{
    if (saved_logger_ && logger_)
    {
        *logger_ = std::move(*saved_logger_);
    }
}

//---------------------------------------------------------------------------//
//! Save a log message
void ScopedLogStorer::operator()(LogProvenance prov,
                                 LogLevel lev,
                                 std::string msg)
{
    static LogLevel const debug_level
        = getenv_loglevel("CELER_LOG_SCOPED", LogLevel::warning);
    if (lev >= debug_level)
    {
        if (getenv_flag("CELER_LOG_SCOPED_VERBOSE", false).value)
        {
            // Print entire message
            debug_clog(prov, lev, msg);
        }
        else
        {
            // Strip colors and only write the first line
            static std::regex const strip_ansi_regex("\033\\[[0-9;]*m");
            auto temp_msg = std::regex_replace(msg, strip_ansi_regex, "");
            auto newline_pos = temp_msg.find_first_of('\n');
            if (newline_pos != std::string::npos)
            {
                // Erase newline and after
                temp_msg.erase(newline_pos);
                temp_msg
                    += "... [truncated: set CELER_LOG_SCOPED_VERBOSE to show]";
            }
            debug_clog(prov, lev, temp_msg);
        }
    }

    if (lev < min_level_)
    {
        return;
    }

    StringSimplifier simplify{float_digits_};
    messages_.push_back(simplify(std::move(msg)));
    levels_.push_back(to_cstring(lev));
}

//---------------------------------------------------------------------------//
//! Delete messages that contain a certain string
void ScopedLogStorer::remove_if_contains(std::string const& s)
{
    auto it_msg = messages_.begin();
    auto it_lev = levels_.begin();

    while (it_msg != messages_.end())
    {
        if (it_msg->find(s) != std::string::npos)
        {
            it_msg = messages_.erase(it_msg);
            it_lev = levels_.erase(it_lev);
        }
        else
        {
            ++it_msg;
            ++it_lev;
        }
    }
}

//---------------------------------------------------------------------------//
//! Print the expected values
void ScopedLogStorer::print_expected() const
{
    using std::cout;
    using std::endl;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static char const* const expected_log_messages[] = "
         << repr(this->messages_)
         << ";\n"
            "EXPECT_VEC_EQ(expected_log_messages, "
            "scoped_log_.messages());\n"
            "static char const* const expected_log_levels[] = "
         << repr(this->levels_)
         << ";\n"
            "EXPECT_VEC_EQ(expected_log_levels, "
            "scoped_log_.levels());\n"
            "/*** END CODE ***/"
         << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print expected results.
 */
std::ostream& operator<<(std::ostream& os, ScopedLogStorer const& logs)
{
    os << "messages: " << repr(logs.messages())
       << "\n"
          "levels: "
       << repr(logs.levels());
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
