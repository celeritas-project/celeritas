//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "corecel/Macros.hh"

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
class VolumeBuilder;
}

//---------------------------------------------------------------------------//
/*!
 * Base class for constructing high-level CSG objects in ORANGE.
 */
class ObjectInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstObject = std::shared_ptr<ObjectInterface const>;
    using VolumeBuilder = detail::VolumeBuilder;
    //!@}

  public:
    //! Short unique name of this object
    virtual std::string_view label() const = 0;

    //! Construct a volume from this object
    virtual NodeId build(VolumeBuilder&) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ObjectInterface() = default;
    virtual ~ObjectInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ObjectInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
