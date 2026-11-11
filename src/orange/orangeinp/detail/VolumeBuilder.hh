//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/cont/InitializedValue.hh"
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
    using Tol = Tolerance<>;
    //!@}

  private:
    // Private types and type aliases
    struct VBTransformPopper;
    using ScopedTransform = InitializedValue<VolumeBuilder*, VBTransformPopper>;

  public:
    // Construct with unit builder (and volume name??)
    explicit VolumeBuilder(CsgUnitBuilder* ub);

    //// ACCESSORS ////

    // Get the construction tolerance
    Tol const& tol() const;

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
    [[nodiscard]] ScopedTransform make_scoped_transform(
        VariantTransform const& t);

  private:
    //// DATA ////

    CsgUnitBuilder* ub_;
    std::vector<TransformId> transforms_;

    //// PRIVATE METHODS ////

    // Add a new variant transform
    void push_transform(VariantTransform&& vt);

    // Pop the last transform, used only by PopVBTransformOnDestruct
    void pop_transform() noexcept(!CELERITAS_DEBUG);
};

//---------------------------------------------------------------------------//
//! Finalizer for ScopedTransform
struct VolumeBuilder::VBTransformPopper
{
    inline void operator()(VolumeBuilder* vb) noexcept(!CELERITAS_DEBUG)
    {
        CELER_EXPECT(vb);
        vb->pop_transform();
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
