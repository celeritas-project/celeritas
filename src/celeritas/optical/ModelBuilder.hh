//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "Types.hh"

namespace celeritas
{
namespace optical
{
class Model;
//---------------------------------------------------------------------------//
struct ModelBuilder
{
    using SPModel = std::shared_ptr<Model>;

    virtual SPModel operator()(ActionId) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
