//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
// Original work:
// https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/blob/7f5d5ec3258d2b7ffbf717e4bd37a3a07285a65f/include/GenericPlacedVolume.h
// Original code from G4VecGeomNav package by John Apostolakis et al.
//---------------------------------------------------------------------------//
//! \file celeritas/ext/g4vg/GenericPlacedVolume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4LogicalVolume.hh>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace g4vg
{
//---------------------------------------------------------------------------//
/*!
 * A generic VecGeom placed volume converted from a Geant4 geometry.
 *
 * Some complex geometry functionalities in Geant4 don't have equivalents yet
 * in VecGeom. This class can be used either as a placeholder, or as a wrapper
 * when it makes sense to use some functionality from the original Geant4
 * volume.
 */
class GenericPlacedVolume : public vecgeom::VPlacedVolume
{
  private:
    //!@{
    //! \name Type aliases
    using Base = vecgeom::VPlacedVolume;
    using LogicalVolume = vecgeom::LogicalVolume;
    using Transformation3D = vecgeom::Transformation3D;
    template<class T>
    using Vector3D = vecgeom::Vector3D<T>;
    template<class T>
    using SOA3D = vecgeom::SOA3D<T>;
    //!@}

  public:
    //! Full constructor, just dispatches to base class constructor.
    GenericPlacedVolume(char const* const label,
                        LogicalVolume const* const logicalVolume,
                        Transformation3D const* const transformation)
        : Base(label, logicalVolume, transformation)
    {
    }

    //! Dispatch to full constructor with a default name, as no name was
    //! provided.
    GenericPlacedVolume(LogicalVolume const* const logicalVolume,
                        Transformation3D const* const transformation)
        : GenericPlacedVolume(
            "GenericPlacedVolume", logicalVolume, transformation)
    {
    }

    //!@{ \name Boilerplate helper functions
    virtual int MemorySize() const override { return sizeof(*this); }
    virtual void PrintType() const override { PrintType(std::cout); }
    virtual void PrintType(std::ostream&) const override {}

    virtual void PrintImplementationType(std::ostream&) const override {}
    virtual void PrintUnplacedType(std::ostream&) const override {}
    //!@}

    //!@{ \name Containing methods
    virtual bool Contains(Vector3D<Precision> const& point) const override
    {
        return GetUnplacedVolume()->Contains(
            GetTransformation()->Transform(point));
    }

    virtual void Contains(SOA3D<Precision> const&, bool* const) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual bool
    Contains(Vector3D<Precision> const&, Vector3D<Precision>&) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual bool
    UnplacedContains(Vector3D<Precision> const& localPoint) const override
    {
        return GetUnplacedVolume()->Contains(localPoint);
    }

    virtual vecgeom::EnumInside
    Inside(Vector3D<Precision> const& point) const override
    {
        return GetUnplacedVolume()->Inside(
            GetTransformation()->Transform(point));
    }

    virtual void
    Inside(SOA3D<Precision> const&, vecgeom::Inside_t* const) const override
    {
    }
    //!@}

    //!@{ \name Geometrical ToIn methods
    virtual Precision
    SafetyToIn(Vector3D<Precision> const& position) const override
    {
        return GetUnplacedVolume()->SafetyToIn(
            GetTransformation()->Transform(position));
    }

    virtual Precision
    DistanceToIn(Vector3D<Precision> const& position,
                 Vector3D<Precision> const& direction,
                 const Precision step_max = vecgeom::kInfLength) const override
    {
        return GetUnplacedVolume()->DistanceToIn(
            GetTransformation()->Transform(position),
            GetTransformation()->TransformDirection(direction),
            step_max);
    }

    virtual void DistanceToIn(SOA3D<Precision> const&,
                              SOA3D<Precision> const&,
                              Precision const* const,
                              Precision* const) const override
    {
    }

    virtual void
    SafetyToIn(SOA3D<Precision> const&, Precision* const) const override
    {
    }

    // if we have any SIMD backend, we offer a SIMD interface
    virtual Real_v
    DistanceToInVec(Vector3D<Real_v> const&,
                    Vector3D<Real_v> const&,
                    const Real_v /*step_max = kInfLength*/) const override
    {
        return Real_v{0.};
    }

    virtual Real_v SafetyToInVec(Vector3D<Real_v> const&) const override
    {
        return Real_v{0.};
    }
    //!@}

    //!@{ \name Geometrical ToOut methods
    VECCORE_ATT_HOST_DEVICE
    virtual Precision
    DistanceToOut(Vector3D<Precision> const& position,
                  Vector3D<Precision> const& direction,
                  Precision const step_max = vecgeom::kInfLength) const override
    {
        return GetUnplacedVolume()->DistanceToOut(
            position, direction, step_max);
    }

    VECCORE_ATT_HOST_DEVICE
    virtual Precision
    PlacedDistanceToOut(Vector3D<Precision> const&,
                        Vector3D<Precision> const&,
                        Precision const /*step_max = kInfLength*/) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual void DistanceToOut(SOA3D<Precision> const&,
                               SOA3D<Precision> const&,
                               Precision const* const,
                               Precision* const) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual void DistanceToOut(SOA3D<Precision> const&,
                               SOA3D<Precision> const&,
                               Precision const* const,
                               Precision* const,
                               int* const) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual Precision
    SafetyToOut(Vector3D<Precision> const& position) const override
    {
        return GetUnplacedVolume()->SafetyToOut(position);
    }

    virtual void
    SafetyToOut(SOA3D<Precision> const&, Precision* const) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    // define this interface in case we don't have the Scalar interface
    virtual Real_v
    DistanceToOutVec(Vector3D<Real_v> const&,
                     Vector3D<Real_v> const&,
                     Real_v const /*step_max = kInfLength*/) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    virtual Real_v SafetyToOutVec(Vector3D<Real_v> const&) const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }
    //!@}

    //! Calculation of surface area.
    virtual Precision SurfaceArea() const override
    {
        return GetUnplacedVolume()->SurfaceArea();
    }

    virtual VPlacedVolume const* ConvertToUnspecialized() const override
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

#ifdef VECGEOM_ROOT
    virtual TGeoShape const* ConvertToRoot() const override { return nullptr; }
#endif

    //! Dispatch to VecGeom function for shape's bounding box in its local
    //! frame.
    VECCORE_ATT_HOST_DEVICE
    virtual void
    Extent(Vector3D<Precision>& min, Vector3D<Precision>& max) const override
    {
        return GetUnplacedVolume()->Extent(min, max);
    }

    //! Dispatch to VecGeom function for shape's normal at `point`.
    VECCORE_ATT_HOST_DEVICE
    virtual bool Normal(Vector3D<Precision> const& point,
                        Vector3D<Precision>& normal) const override
    {
        return GetUnplacedVolume()->Normal(
            GetTransformation()->Transform(point), normal);
    }

    //! Dispatch to VecGeom function to compute shape's volumetric capacity.
    virtual Precision Capacity() override
    {
        return GetUnplacedVolume()->Capacity();
    }

#ifdef VECGEOM_CUDA_INTERFACE
    using CudaVPlacedVolume = vecgeom::cuda::VPlacedVolume;
    using CudaLogicalVolume = vecgeom::cuda::LogicalVolume;
    using CudaTransformation3D = vecgeom::cuda::Transformation3D;
    template<class T>
    using DevicePtr = vecgeom::DevicePtr<T>;

    //! @{ /name Required interface
    //
    // These implementations are required when CUDA is enabled.
    // A trivialimplementation is okay, since won't be called from GPU.
    virtual size_t DeviceSizeOf() const override { return 0; }

    DevicePtr<CudaVPlacedVolume>
    CopyToGpu(DevicePtr<CudaLogicalVolume> const,
              DevicePtr<CudaTransformation3D> const,
              DevicePtr<CudaVPlacedVolume> const) const
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    DevicePtr<CudaVPlacedVolume>
    CopyToGpu(DevicePtr<CudaLogicalVolume> const,
              DevicePtr<CudaTransformation3D> const) const
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }

    void CopyManyToGpu(std::vector<VPlacedVolume const*> const&,
                       std::vector<DevicePtr<CudaLogicalVolume>> const&,
                       std::vector<DevicePtr<CudaTransformation3D>> const&,
                       std::vector<DevicePtr<CudaVPlacedVolume>> const&) const
    {
        CELER_NOT_IMPLEMENTED("GenericPlacedVolume");
    }
    //!@}
#endif  // VECGEOM_CUDA_INTERFACE

  private:
    //!@{ \name Deleted constructors and assignment operator.
    ~GenericPlacedVolume() = default;
    GenericPlacedVolume(GenericPlacedVolume const&) = delete;
    GenericPlacedVolume(GenericPlacedVolume const&&) = delete;
    GenericPlacedVolume& operator=(GenericPlacedVolume const&) = delete;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
