//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <iostream>
#include <sstream>
#include "base/ColorUtils.hh"
#include "Communicator.hh"

namespace
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
//! Default global logger prints the error message with basic colors
void default_global_handler(Provenance prov, LogLevel lev, std::string msg)
{
    if (lev == LogLevel::debug || lev >= LogLevel::warning)
    {
        // Output problem line/file for debugging or high level
        std::cerr << color_code('x') << prov.file << ':' << prov.line
                  << color_code(' ') << ": ";
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
    };
    // clang-format on
    std::cerr << color_code(c) << to_cstring(lev) << ": " << color_code(' ')
              << msg << std::endl;
}

//---------------------------------------------------------------------------//
//! Log the local node number as well as the message
class LocalHandler
{
  public:
    explicit LocalHandler(const Communicator& comm) : rank_(comm.rank()) {}

    void operator()(Provenance prov, LogLevel lev, std::string msg)
    {
        // To avoid multiple process output stepping on each other, write into
        // a buffer and then print with a single call.
        std::ostringstream os;
        os << color_code('x') << prov.file << ':' << prov.line
           << color_code(' ') << ": " << color_code('W') << "rank " << rank_
           << ": " << color_code('x') << to_cstring(lev) << ": "
           << color_code(' ') << msg << '\n';
        std::cerr << os.str();
    }

  private:
    int rank_;
};

//---------------------------------------------------------------------------//
} // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with communicator (only rank zero is active) and handler.
 */
Logger::Logger(const Communicator& comm, LogHandler handle)
{
    if (comm.rank() == 0)
    {
        // Accept handler, otherwise it is a "null" function pointer.
        handle_ = std::move(handle);
    }
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Parallel-enabled logger: print only on "main" process.
 */
Logger& world_logger()
{
    static Logger logger(Communicator::comm_world(), &default_global_handler);
    return logger;
}

//---------------------------------------------------------------------------//
/*!
 * Serial logger: print on *every* process that calls it.
 */
Logger& self_logger()
{
    static Logger logger(Communicator::comm_self(),
                         LocalHandler{Communicator::comm_world()});
    return logger;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
