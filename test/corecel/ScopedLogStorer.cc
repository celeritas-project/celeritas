//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/ScopedLogStorer.cc
//---------------------------------------------------------------------------//
#include "ScopedLogStorer.hh"

#include <iostream>
#include <regex>

#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct reference to log to temporarily replace.
 */
ScopedLogStorer::ScopedLogStorer(Logger* orig, LogLevel min_level)
    : logger_{orig}
{
    CELER_EXPECT(logger_);
    CELER_EXPECT(min_level != LogLevel::size_);
    // Create a new logger that calls our operator(), replace orig and store
    saved_logger_ = std::make_unique<Logger>(
        std::exchange(*logger_, Logger{std::ref(*this)}));
    // Update global log level
    logger_->level(min_level);
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
    if (saved_logger_)
    {
        *logger_ = std::move(*saved_logger_);
    }
}

//---------------------------------------------------------------------------//
//! Save a log message
void ScopedLogStorer::operator()(Provenance, LogLevel lev, std::string msg)
{
    static const std::regex delete_ansi("\033\\[[0-9;]*m");
    static const std::regex subs_ptr("0x[0-9a-f]+");
    msg = std::regex_replace(msg, delete_ansi, "");
    msg = std::regex_replace(msg, subs_ptr, "0x0");
    messages_.push_back(std::move(msg));
    levels_.push_back(to_cstring(lev));
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
