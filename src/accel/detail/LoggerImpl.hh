//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LoggerImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string_view>

#include "corecel/io/LoggerTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Stream wrapper for writing an optionally abbreviated filename to output
struct CleanedProvenance
{
    std::string_view filename;
    int line{};
};

// Write abbreviated filename to output
std::ostream& operator<<(std::ostream&, CleanedProvenance);

//---------------------------------------------------------------------------//
//! Stream wrapper for color-annotated message
struct ColorfulLogMessage
{
    LogProvenance const& prov;
    LogLevel lev;
    std::string const& msg;
};

// Write color-annotated message
std::ostream& operator<<(std::ostream&, ColorfulLogMessage);

//---------------------------------------------------------------------------//
/*!
 * Write the Geant4 thread ID on output.
 */
class MtSelfWriter
{
  public:
    // Construct with number of threads
    explicit MtSelfWriter(int num_threads);

    // Write output with thread ID
    void operator()(LogProvenance prov, LogLevel lev, std::string msg) const;

  private:
    int num_threads_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
