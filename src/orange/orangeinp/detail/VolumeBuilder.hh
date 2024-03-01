//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Label.hh"
#include "orange/transform/VariantTransform.hh"

#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
class CsgUnitBuilder;
struct BoundingZone;
class PopVBTransformOnDestruct;

//---------------------------------------------------------------------------//
/*!
 * Construct volumes out of objects.
 *
 * This class maintains a stack of transforms used by nested objects. It
 * ultimately returns a node ID corresponding to the CSG node (and bounding box
 * etc.) of the constructed object.
 *
 * To add a transform, store the result of \c make_scoped_transform within a
 * scoping block (generally the calling function). The resulting RAII class
 * prevents an imbalance from manual calls to "push" and "pop". For an example
 * usage, see \c celeritas::orangeinp::Transformed::build .
 */
class VolumeBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Metadata = Label;
    //!@}

  public:
    // Construct with unit builder (and volume name??)
    explicit VolumeBuilder(CsgUnitBuilder* ub);

    //// ACCESSORS ////

    //!@{
    //! Access the unit builder for construction
    CsgUnitBuilder const& unit_builder() const { return *ub_; }
    CsgUnitBuilder& unit_builder() { return *ub_; }
    //!@}

    // Access the local-to-global transform during construction
    VariantTransform const& local_transform() const;

    //// MUTATORS ////

    // Add a region to the CSG tree, automatically calculating bounding zone
    NodeId insert_region(Metadata&& md, Joined&& j);

    // Add a region to the CSG tree, including a better bounding zone
    NodeId insert_region(Metadata&& md, Joined&& j, BoundingZone&& bz);

    // Add a negated region to the CSG tree
    NodeId insert_region(Metadata&& md, Negated&& n);

    // Apply a transform within this scope
    [[nodiscard]] PopVBTransformOnDestruct
    make_scoped_transform(VariantTransform const& t);

  private:
    //// DATA ////

    CsgUnitBuilder* ub_;
    std::vector<TransformId> transforms_;

    //// PRIVATE METHODS ////

    // Add a new variant transform
    void push_transform(VariantTransform&& vt);

    // Pop the last transform, used only by PopVBTransformOnDestruct
    void pop_transform();

    //// FRIENDS ////

    friend class PopVBTransformOnDestruct;
};

//---------------------------------------------------------------------------//
//! Implementation-only RAII helper class for VolumeBuilder (detail detail)
class PopVBTransformOnDestruct
{
  private:
    friend class VolumeBuilder;

    // Construct with a volume builder pointer
    explicit PopVBTransformOnDestruct(VolumeBuilder* vb);

  public:
    //! Capture the pointer when move constructed
    PopVBTransformOnDestruct(PopVBTransformOnDestruct&& other)
        : vb_(std::exchange(other.vb_, nullptr))
    {
    }

    //! Capture the pointer when move assigned
    PopVBTransformOnDestruct& operator=(PopVBTransformOnDestruct&& other)
    {
        vb_ = std::exchange(other.vb_, nullptr);
        return *this;
    }

    //! Call pop when we own the pointer and go out of scope
    ~PopVBTransformOnDestruct()
    {
        if (vb_)
        {
            vb_->pop_transform();
        }
    }

  private:
    VolumeBuilder* vb_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
