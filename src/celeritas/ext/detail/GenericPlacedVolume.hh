//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file GenericPlacedVolume.hh
 * \brief Class for a generic placed volume related to G4 to VecGeom conversion
 *
 * Original code from G4VecGeomNav package by John Apostolakis et.al.
 *
 * Original source:
 *   https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/blob/7f5d5ec3258d2b7ffbf717e4bd37a3a07285a65f/include/GenericPlacedVolume.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <G4LogicalVolume.hh>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>
using namespace vecgeom;

namespace celeritas
{

#define VECGEOM_VECTORAPI

class GenericPlacedVolume : public vecgeom::VPlacedVolume
{
  public:
    using Base = vecgeom::VPlacedVolume;
    // using Base::Base;
    GenericPlacedVolume(char const* const             label,
                        LogicalVolume const* const    logicalVolume,
                        Transformation3D const* const transformation)
        : Base(label, logicalVolume, transformation)
    {
    }

    GenericPlacedVolume(LogicalVolume const* const    logicalVolume,
                        Transformation3D const* const transformation)
        : GenericPlacedVolume("", logicalVolume, transformation)
    {
    }

    virtual int  MemorySize() const override { return sizeof(*this); }
    virtual void PrintType() const override { PrintType(std::cout); }
    virtual void PrintType(std::ostream&) const override {}

    virtual void PrintImplementationType(std::ostream&) const override {}
    virtual void PrintUnplacedType(std::ostream&) const override {}
    virtual bool Contains(Vector3D<Precision> const& point) const override
    {
        return GetUnplacedVolume()->Contains(
            GetTransformation()->Transform(point));
    }
    virtual void Contains(SOA3D<Precision> const&, bool* const) const override
    {
    }
    virtual bool
    Contains(Vector3D<Precision> const&, Vector3D<Precision>&) const override
    {
        assert(false);
        return false;
    }

    virtual bool
    UnplacedContains(Vector3D<Precision> const& localPoint) const override
    {
        return GetUnplacedVolume()->Contains(localPoint);
    }

    virtual EnumInside Inside(Vector3D<Precision> const& point) const override
    {
        return GetUnplacedVolume()->Inside(
            GetTransformation()->Transform(point));
    }

    virtual void Inside(SOA3D<Precision> const&, Inside_t* const) const override
    {
    }

    virtual Precision
    SafetyToIn(Vector3D<Precision> const& position) const override
    {
        return GetUnplacedVolume()->SafetyToIn(
            GetTransformation()->Transform(position));
    }

    virtual Precision
    DistanceToIn(Vector3D<Precision> const& position,
                 Vector3D<Precision> const& direction,
                 const Precision step_max = kInfLength) const override
    {
        return GetUnplacedVolume()->DistanceToIn(
            GetTransformation()->Transform(position),
            GetTransformation()->TransformDirection(direction),
            step_max);
    }

#ifdef VECGEOM_VECTORAPI
    // if we have any SIMD backend, we offer a SIMD interface
    virtual Real_v
    DistanceToInVec(Vector3D<Real_v> const&,
                    Vector3D<Real_v> const&,
                    const Real_v /*step_max = kInfLength*/) const override
    {
        return Real_v{0.};
    }
#endif

    virtual void DistanceToIn(SOA3D<Precision> const&,
                              SOA3D<Precision> const&,
                              Precision const* const,
                              Precision* const) const override
    {
    }

    VECCORE_ATT_HOST_DEVICE
    virtual Precision
    DistanceToOut(Vector3D<Precision> const& position,
                  Vector3D<Precision> const& direction,
                  Precision const step_max = kInfLength) const override
    {
        return GetUnplacedVolume()->DistanceToOut(
            position, direction, step_max);
    }

#ifdef VECGEOM_VECTORAPI
    // define this interface in case we don't have the Scalar interface
    virtual Real_v
    DistanceToOutVec(Vector3D<Real_v> const&,
                     Vector3D<Real_v> const&,
                     Real_v const /*step_max = kInfLength*/) const override
    {
        return Real_v{0.};
    }
#endif

    VECCORE_ATT_HOST_DEVICE
    virtual Precision
    PlacedDistanceToOut(Vector3D<Precision> const&,
                        Vector3D<Precision> const&,
                        Precision const /*step_max = kInfLength*/) const override
    {
        assert(false);
        return 0;
    }

    virtual void DistanceToOut(SOA3D<Precision> const&,
                               SOA3D<Precision> const&,
                               Precision const* const,
                               Precision* const) const override
    {
    }

    virtual void DistanceToOut(SOA3D<Precision> const&,
                               SOA3D<Precision> const&,
                               Precision const* const,
                               Precision* const,
                               int* const) const override
    {
    }

#ifdef VECGEOM_VECTORAPI
    virtual Real_v SafetyToInVec(Vector3D<Real_v> const&) const override
    {
        return Real_v{0.};
    }
#endif

    virtual void
    SafetyToIn(SOA3D<Precision> const&, Precision* const) const override
    {
    }

    virtual Precision
    SafetyToOut(Vector3D<Precision> const& position) const override
    {
        return GetUnplacedVolume()->SafetyToOut(position);
    }

#ifdef VECGEOM_VECTORAPI
    virtual Real_v SafetyToOutVec(Vector3D<Real_v> const&) const override
    {
        return Real_v{0.};
    }
#endif

    virtual void
    SafetyToOut(SOA3D<Precision> const&, Precision* const) const override
    {
    }

    virtual Precision SurfaceArea() const override
    {
        return GetUnplacedVolume()->SurfaceArea();
    }

    virtual VPlacedVolume const* ConvertToUnspecialized() const override
    {
        return nullptr;
    }

#ifdef VECGEOM_ROOT
    virtual TGeoShape const* ConvertToRoot() const override
    {
        return nullptr;
    }
#endif

    VECCORE_ATT_HOST_DEVICE
    virtual void
    Extent(Vector3D<Precision>& min, Vector3D<Precision>& max) const override
    {
        return GetUnplacedVolume()->Extent(min, max);
    }

    VECCORE_ATT_HOST_DEVICE
    virtual bool Normal(Vector3D<Precision> const& point,
                        Vector3D<Precision>&       normal) const override
    {
        return GetUnplacedVolume()->Normal(
            GetTransformation()->Transform(point), normal);
    }

    virtual Precision Capacity() override
    {
        return GetUnplacedVolume()->Capacity();
    }

#ifdef VECGEOM_CUDA_INTERFACE
    virtual size_t DeviceSizeOf() const override
    {
        return 0;
    }

    DevicePtr<cuda::VPlacedVolume>
    CopyToGpu(DevicePtr<cuda::LogicalVolume> const,
              DevicePtr<cuda::Transformation3D> const,
              DevicePtr<cuda::VPlacedVolume> const) const
    {
        return {};
    }

    DevicePtr<cuda::VPlacedVolume>
    CopyToGpu(DevicePtr<cuda::LogicalVolume> const,
              DevicePtr<cuda::Transformation3D> const) const
    {
        return {};
    }

    void CopyManyToGpu(std::vector<VPlacedVolume const*> const&,
                       std::vector<DevicePtr<cuda::LogicalVolume>> const&,
                       std::vector<DevicePtr<cuda::Transformation3D>> const&,
                       std::vector<DevicePtr<cuda::VPlacedVolume>> const&) const
    {
    }
#endif  // VECGEOM_CUDA_INTERFACE
};
}  // namespace celeritas
