//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store surface data corresponding to a volume.
 *
 * This stores information about the surfaces (both boundary and interface)
 * of a volume. The \em boundary is a single optional surface ID, and the
 * \em interface is an unzipped map \code (pre, post) -> surface \endcode.
 *
 * If \c interface_pre and \c interface_post are zipped, the result is \em
 * sorted.  In other words, the pre-step surface can be searched with
 * bisection, and the resulting subrange can also be searched with bisection to
 * find the post-step surface. This then corresponds to the \c SurfaceId of
 * that interface.
 */
struct VolumeSurfaceRecord
{
    //! Surface identifier for the volume boundary
    SurfaceId boundary;

    //! Sorted range of exiting volume instances (from this volume)
    ItemRange<VolumeInstanceId> interface_pre;

    //! Corresponding range of entering volume instances (to other volumes)
    ItemRange<VolumeInstanceId> interface_post;

    //! Surface IDs for the pre->post mapping
    ItemRange<SurfaceId> surface;

    //! True if valid data is present
    explicit CELER_FUNCTION operator bool() const
    {
        return boundary
               || (!interface_pre.empty()
                   && interface_pre.size() == interface_post.size()
                   && interface_pre.size() == surface.size());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data for mapping between volumes and their surfaces.
 *
 * This structure stores device-compatible data relating volumes and their
 * surfaces, primarily for optical physics at material interfaces.
 *
 * If no "interface" surfaces are present then the backend storage arrays will
 * be empty.
 *
 * \todo We might want to have separate arrays for boundary and interface
 * surfaces since models typically have one or the other but not both, and it
 * could potentially reduce memory access requirements.
 */
template<Ownership W, MemSpace M>
struct SurfaceParamsData
{
    //// TYPES ////

    template<class T>
    using VolumeItems = Collection<T, W, M, VolumeId>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Number of surfaces
    SurfaceId::size_type num_surfaces{0};

    //! Surface properties for logical volumes
    VolumeItems<VolumeSurfaceRecord> volume_surfaces;

    //! Backend storage for PV->PV mapping
    Items<VolumeInstanceId> volume_instance_ids;

    //! Backend storage for surface interfaces
    Items<SurfaceId> surface_ids;

    //// METHODS ////

    //! True if surfaces are present
    explicit CELER_FUNCTION operator bool() const { return num_surfaces != 0; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfaceParamsData& operator=(SurfaceParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        CELER_EXPECT(!other.volume_surfaces.empty());

        num_surfaces = other.num_surfaces;
        volume_surfaces = other.volume_surfaces;
        volume_instance_ids = other.volume_instance_ids;
        surface_ids = other.surface_ids;

        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
