//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Truncated.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "IntersectRegion.hh"
#include "ObjectInterface.hh"
#include "Shape.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * A shape formed by truncating another region with axis-aligned planes.
 *
 * Geant4 uses this for the ellipsoid along \em -/+z and SCALE uses this for
 * chords along all three axes.
 */
class Truncated final : public ObjectInterface
{
  public:
    using Plane = InfPlane;
    using VecPlane = std::vector<Plane>;
    using UPRegion = std::unique_ptr<IntersectRegionInterface>;

  public:
    // Return a truncated *or* shape given optional planes
    template<class T>
    inline static SPConstObject
    or_shape(std::string&& label, T&& interior, VecPlane&& truncated);

    // Construct with a region to truncate and a vector of planes
    Truncated(std::string&& label, UPRegion&& region, VecPlane&& planes);

    // Build the volume using the volume builder
    NodeId build(VolumeBuilder& vb) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Get the label for this object
    std::string_view label() const final { return label_; }

    //! Get the truncated region
    IntersectRegionInterface const& region() const { return *region_; }

    //! Get the truncating planes
    VecPlane const& planes() const { return planes_; }

  private:
    std::string label_;
    std::unique_ptr<IntersectRegionInterface> region_;
    VecPlane planes_;
};

//---------------------------------------------------------------------------//
/*!
 * Create a truncated region or just a shape.
 */
template<class T>
auto Truncated::or_shape(std::string&& label,
                         T&& interior,
                         VecPlane&& truncated) -> SPConstObject
{
    if (truncated.empty())
    {
        // Just a shape
        return std::make_shared<Shape<T>>(std::move(label),
                                          std::forward<T>(interior));
    }
    return std::make_shared<Truncated>(
        std::move(label),
        std::make_unique<T>(std::forward<T>(interior)),
        std::move(truncated));
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
