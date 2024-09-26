//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class Model;

//---------------------------------------------------------------------------//
/*!
 * Abstract base class used to build optical models.
 */
struct ModelBuilder
{
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<Model>;
    //!@}

    //! Construct model with given action identifer
    virtual SPModel operator()(ActionId id) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
