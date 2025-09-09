//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootJsonDumper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Write an \c ImportData object to JSON output.
 *
 * \code
 *  RootJsonDumper dump(&std::cout);
 *  dump(my_import_data);
 * \endcode
 */
class RootJsonDumper
{
  public:
    // Construct with an output stream
    explicit RootJsonDumper(std::ostream& os);

    // Save data to the JSON file
    void operator()(ImportData const& data);

  private:
    std::ostream* os_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootJsonDumper::RootJsonDumper(std::ostream&)
{
    CELER_DISCARD(os_);
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootJsonDumper::operator()(ImportData const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
