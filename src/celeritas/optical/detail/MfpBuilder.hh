//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/MfpBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/grid/GenericGridInserter.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    MfpBuilder ...;
   \endcode
 */
class MfpBuilder
{
  public:
    using GenericGridId = OpaqueId<GenericGridRecord>;

    MfpBuilder(GenericGridInserter<GenericGridId>* inserter,
               std::vector<GenericGridId>* sink)
        : insert_(inserter), ids_(sink)
    {
        CELER_EXPECT(insert_);
        CELER_EXPECT(ids_);
    }

    template<typename... Args>
    void operator()(Args const&... args)
    {
        ids_->push_back((*insert_)(args...));
    }

  private:
    GenericGridInserter<GenericGridId>* insert_;
    std::vector<GenericGridId>* ids_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
