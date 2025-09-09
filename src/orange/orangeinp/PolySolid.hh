//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/PolySolid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/OpaqueId.hh"

#include "IntersectRegion.hh"
#include "ObjectInterface.hh"
#include "Solid.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Radial extents and axial segments for a stacked solid.
 *
 * Axial grid points must be nondecreasing. If "inner" points are specified,
 * they must be less than the outer points and more than zero. The inner list
 * is allowed to be empty indicating no inner (hollow) exclusion.
 */
class PolySegments
{
  public:
    //!@{
    //! \name Type aliases
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct from a filled polygon solid
    PolySegments(VecReal&& outer, VecReal&& z);

    // Construct from a shell of a polygon solid
    PolySegments(VecReal&& inner, VecReal&& outer, VecReal&& z);

    //! Number of segments (one less than grid points)
    size_type size() const { return outer_.size() - 1; }

    // Access the inner radii (for building 'exclusion' shape)
    inline VecReal const& inner() const;

    //! Access the outer radii (for building 'interior' shape)
    VecReal const& outer() const { return outer_; }

    //! Access the z planes
    VecReal const& z() const { return z_; }

    // Access lo/hi inner/exclusion radii for a segment
    inline Real2 inner(size_type) const;

    // Access lo/hi outer radii for a segment
    inline Real2 outer(size_type) const;

    // Access lo/hi z values for a segment
    inline Real2 z(size_type) const;

    //! Whether there is an internal subtraction from the poly
    bool has_exclusion() const { return !inner_.empty(); }

  private:
    VecReal inner_;
    VecReal outer_;
    VecReal z_;
};

//---------------------------------------------------------------------------//
/*!
 * Access the inner radii (for building 'exclusion' shape).
 */
auto PolySegments::inner() const -> VecReal const&
{
    CELER_EXPECT(has_exclusion());
    return inner_;
}

//---------------------------------------------------------------------------//
/*!
 * Access lo/hi inner/exclusion radii for a segment.
 */
auto PolySegments::inner(size_type i) const -> Real2
{
    CELER_EXPECT(this->has_exclusion() && i < this->size());
    return {inner_[i], inner_[i + 1]};
}

//---------------------------------------------------------------------------//
/*!
 * Access lo/hi outer radii for a segment.
 */
auto PolySegments::outer(size_type i) const -> Real2
{
    CELER_EXPECT(i < this->size());
    return {outer_[i], outer_[i + 1]};
}

//---------------------------------------------------------------------------//
/*!
 * Access lo/hi z values for a segment.
 */
auto PolySegments::z(size_type i) const -> Real2
{
    CELER_EXPECT(i < this->size());
    return {z_[i], z_[i + 1]};
}

//---------------------------------------------------------------------------//
/*!
 * A segmented stack of same-type shapes with an azimuthal truncation.
 */
class PolySolidBase : public ObjectInterface
{
  public:
    // Anchored default virtual destructor
    ~PolySolidBase() override;

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Axial segments
    PolySegments const& segments() const { return segments_; }

    //! Optional azimuthal angular restriction
    EnclosedAzi enclosed_azi() const { return enclosed_; }

  protected:
    PolySolidBase(std::string&& label,
                  PolySegments&& segments,
                  EnclosedAzi&& enclosed);

    //!@{
    //! Allow construction and assignment only through daughter classes
    CELER_DEFAULT_COPY_MOVE(PolySolidBase);
    //!@}

  private:
    std::string label_;
    PolySegments segments_;
    EnclosedAzi enclosed_;
};

//---------------------------------------------------------------------------//
/*!
 * A series of stacked cones or cylinders or combination of both.
 */
class PolyCone final : public PolySolidBase
{
  public:
    // Return a polycone *or* a simplified version for only a single segment
    static SPConstObject or_solid(std::string&& label,
                                  PolySegments&& segments,
                                  EnclosedAzi&& enclosed);

    // Build with label, axial segments, optional restriction
    PolyCone(std::string&& label,
             PolySegments&& segments,
             EnclosedAzi&& enclosed);

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the shape to JSON
    void output(JsonPimpl*) const final;
};

//---------------------------------------------------------------------------//
/*!
 * A series of stacked regular prisms or cone-y prisms.
 */
class PolyPrism final : public PolySolidBase
{
  public:
    // Return a polyprism *or* a simplified version for only a single segment
    static SPConstObject or_solid(std::string&& label,
                                  PolySegments&& segments,
                                  EnclosedAzi&& enclosed,
                                  int num_sides,
                                  real_type orientation);

    // Build with label, axial segments, parameters, optional restriction
    PolyPrism(std::string&& label,
              PolySegments&& segments,
              EnclosedAzi&& enclosed,
              int num_sides,
              real_type orientation);

    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the shape to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Number of sides
    int num_sides() const { return num_sides_; }
    //! Rotation factor
    real_type orientation() const { return orientation_; }

  private:
    int num_sides_;
    real_type orientation_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
