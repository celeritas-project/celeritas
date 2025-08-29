//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <vector>

#include "PolishedRoughnessExecutor.hh"

namespace celeritas
{
namespace inp
{
struct NoRoughness;
}
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    PolishedRoughnessModel ...;
   \endcode
 */
class PolishedRoughnessModelController
{
  public:
    constexpr static std::string_view label = "polished";

    PolishedRoughnessModelController(std::vector<inp::NoRoughness> const&) {}

    template<MemSpace M>
    PolishedRoughnessExecutorBuilder make_builder() const
    {
        return PolishedRoughnessExecutorBuilder{};
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
