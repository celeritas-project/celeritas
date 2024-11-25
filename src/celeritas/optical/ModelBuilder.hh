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
/*!
 * Concrete base class for deferred building of optical models.
 *
 * Mimics \c Process in core physics, but doesn't need any functionality
 * besides building the models.
 */
struct ModelBuilder
{
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<Model>;
    //!@}

    //! Construct an optical model with the given action ID
    virtual SPModel operator()(ActionId) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
