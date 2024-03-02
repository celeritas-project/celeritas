//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Transformed.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/transform/VariantTransform.hh"

#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Build a translated or transformed object.
 */
class Transformed final : public ObjectInterface
{
  public:
    // Construct with daughter object and transform
    Transformed(SPConstObject obj, VariantTransform&& transform);

    //! Get the user-provided label
    std::string_view label() const final { return obj_->label(); }

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

  private:
    SPConstObject obj_;
    VariantTransform transform_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
