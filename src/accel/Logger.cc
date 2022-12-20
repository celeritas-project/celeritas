//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4ios.hh>

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/MpiCommunicator.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
class MtLogger
{
  public:
    explicit MtLogger(int num_threads);
    void operator()(Provenance prov, LogLevel lev, std::string msg);

  private:
    int num_threads_;
};

//---------------------------------------------------------------------------//
MtLogger::MtLogger(int num_threads) : num_threads_(num_threads)
{
    CELER_EXPECT(num_threads_ > 0);
}

//---------------------------------------------------------------------------//
void MtLogger::operator()(Provenance prov, LogLevel lev, std::string msg)
{
    auto& cerr = G4cerr;
    {
        auto last_slash = std::find(prov.file.rbegin(), prov.file.rend(), '/');
        if (!prov.file.empty() && last_slash == prov.file.rend())
        {
            --last_slash;
        }

        // Output problem line/file for debugging or high level
        cerr << color_code('x')
             << std::string(last_slash.base(), prov.file.end());
        if (prov.line)
            cerr << ':' << prov.line;
        cerr << color_code(' ') << ": ";
    }

    int local_thread = G4Threading::G4GetThreadId();
    if (local_thread >= 0)
    {
        // Threading is initialized
        cerr << color_code('W') << '[' << G4Threading::G4GetThreadId() << '/'
             << num_threads_ << "] " << color_code('x');
    }

    // clang-format off
    char c = ' ';
    switch (lev)
    {
        case LogLevel::debug:      c = 'x'; break;
        case LogLevel::diagnostic: c = 'x'; break;
        case LogLevel::status:     c = 'b'; break;
        case LogLevel::info:       c = 'g'; break;
        case LogLevel::warning:    c = 'y'; break;
        case LogLevel::error:      c = 'r'; break;
        case LogLevel::critical:   c = 'R'; break;
        case LogLevel::size_: CELER_ASSERT_UNREACHABLE();
    };
    // clang-format on
    cerr << color_code(c) << to_cstring(lev) << ": " << color_code(' ') << msg
         << std::endl;
}
//---------------------------------------------------------------------------//
} // namespace

Logger make_mt_logger(const G4RunManager& runman)
{
    return Logger(MpiCommunicator{},
                  MtLogger{runman.GetNumberOfThreads()},
                  "CELER_LOG_LOCAL");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
