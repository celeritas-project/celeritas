//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "corecel/Macros.hh"

#include "CsgTypes.hh"

namespace celeritas
{
struct JsonPimpl;

namespace orangeinp
{
//---------------------------------------------------------------------------//
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
    // Anchored default destructor
    virtual ~ObjectInterface() = 0;

    //! Short unique name of this object
    virtual std::string_view label() const = 0;

    //! Construct a volume from this object
    virtual NodeId build(VolumeBuilder&) const = 0;

    //! Write the region to a JSON object
    virtual void output(JsonPimpl*) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ObjectInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ObjectInterface);
    //!@}
};

//---------------------------------------------------------------------------//
// Get a JSON string representing an object
std::string to_string(ObjectInterface const&);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
