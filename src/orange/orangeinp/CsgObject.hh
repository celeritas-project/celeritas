//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Everywhere *but* the embedded object.
 */
class NegatedObject : public ObjectInterface
{
  public:
    // Construct with the object to negate and an empty name
    explicit NegatedObject(SPConstObject obj);

    // Construct with a name and object
    NegatedObject(std::string&& label, SPConstObject obj);

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

  private:
    std::string label_;
    SPConstObject obj_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
