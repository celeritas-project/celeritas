//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file GenericSolid.hh
 * \brief Class for a generic solid related to G4 to VecGeom conversion
 *
 * Original code from G4VecGeomNav package by John Apostolakis et.al.
 *
 * Original source:
 *   https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/ce4a0bb2f777a6728e59dee96c764fd61fa8b785/include/VecGeomG4Solid.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include "GenericPlacedVolume.hh"
#define VECGEOM_VECTORAPI

#include <iostream>

#include "Geant4/G4VSolid.hh"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"

template<typename S>
class GenericSolid : public vecgeom::VUnplacedVolume
{
  public:
    using vecgeom::VUnplacedVolume::DistanceToIn;
    using vecgeom::VUnplacedVolume::DistanceToOut;
    using vecgeom::VUnplacedVolume::SafetyToIn;
    using vecgeom::VUnplacedVolume::SafetyToOut;

    explicit GenericSolid(S const* g4solid) : fG4Solid(g4solid) {}

    static G4ThreeVector ToG4V(vecgeom::Vector3D<double> const& p)
    {
        return G4ThreeVector(p[0], p[1], p[2]);
    }

    static EnumInside ConvertEnum(::EInside p)
    {
        if (p == ::EInside::kInside)
            return EnumInside::kInside;
        if (p == ::EInside::kSurface)
            return EnumInside::kSurface;
        return EnumInside::kOutside;
    }

    // ---------------- Contains
    // --------------------------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    bool Contains(vecgeom::Vector3D<Precision> const& p) const override
    {
        auto a = ConvertEnum(fG4Solid->Inside(ToG4V(p)));
        if (a == EnumInside::kOutside)
            return false;
        return true;
    }

    VECCORE_ATT_HOST_DEVICE
    EnumInside Inside(vecgeom::Vector3D<Precision> const& p) const override
    {
        return ConvertEnum(fG4Solid->Inside(ToG4V(p)));
    }

    // ---------------- DistanceToOut functions
    // -----------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision DistanceToOut(vecgeom::Vector3D<Precision> const& p,
                            vecgeom::Vector3D<Precision> const& d,
                            Precision /*step_max = kInfLength*/) const override
    {
        return fG4Solid->DistanceToOut(ToG4V(p), ToG4V(d)); // , bool
                                                            // calculateNorm =
                                                            // false);
    }

    // the USolid/GEANT4-like interface for DistanceToOut (returning also
    // exiting normal)
    VECCORE_ATT_HOST_DEVICE
    Precision DistanceToOut(vecgeom::Vector3D<Precision> const& p,
                            vecgeom::Vector3D<Precision> const& d,
                            vecgeom::Vector3D<Precision>&       normal,
                            bool&                               convex,
                            Precision /*step_max = kInfLength*/) const override
    {
        bool          calculateNorm = true;
        G4ThreeVector normalG4;
        auto          dist = fG4Solid->DistanceToOut(
            ToG4V(p), ToG4V(d), calculateNorm, &convex, &normalG4);
        normal = vecgeom::Vector3D<vecgeom::Precision>(
            normalG4[0], normalG4[1], normalG4[2]);
        return dist;
    }

    // ---------------- SafetyToOut functions
    // -----------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision SafetyToOut(vecgeom::Vector3D<Precision> const& p) const override
    {
        return fG4Solid->DistanceToOut(ToG4V(p));
    }

    //#ifdef VECGEOM_VECTORAPI
    // an explicit SIMD interface
    // VECCORE_ATT_HOST_DEVICE
    // Real_v SafetyToOutVec(vecgeom::Vector3D<Real_v> const &p) const override
    // {
    //   return Real_v(0);
    // }
    // #endif

    // ---------------- DistanceToIn functions
    // -----------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision
    DistanceToIn(vecgeom::Vector3D<Precision> const& p,
                 vecgeom::Vector3D<Precision> const& d,
                 const Precision /*step_max = kInfLength*/) const override
    {
        return fG4Solid->DistanceToIn(ToG4V(p), ToG4V(d));
    }

    // ---------------- SafetyToIn functions
    // -------------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision SafetyToIn(vecgeom::Vector3D<Precision> const& p) const override
    {
        return fG4Solid->DistanceToIn(ToG4V(p));
    }

    // ---------------- Normal
    // ---------------------------------------------------------------------

    VECCORE_ATT_HOST_DEVICE
    bool Normal(vecgeom::Vector3D<Precision> const& p,
                vecgeom::Vector3D<Precision>&       normal) const override
    {
        auto n = fG4Solid->SurfaceNormal(ToG4V(p));
        normal = vecgeom::Vector3D<double>(n[0], n[1], n[2]);
        return true;
    }

    // ----------------- Extent
    // --------------------------------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    void Extent(vecgeom::Vector3D<Precision>& aMin,
                vecgeom::Vector3D<Precision>& aMax) const override
    {
        auto ext = fG4Solid->GetExtent();
        aMin.Set(ext.GetXmin(), ext.GetYmin(), ext.GetZmin());
        aMax.Set(ext.GetXmax(), ext.GetYmax(), ext.GetZmax());
    }

    double Capacity() const override
    {
        return const_cast<S*>(fG4Solid)->S::GetCubicVolume();
    }

    double SurfaceArea() const override
    {
        return const_cast<S*>(fG4Solid)->S::GetSurfaceArea();
    }

    int  MemorySize() const override { return sizeof(this); }
    void Print(std::ostream& os) const override
    {
        if (fG4Solid)
        {
            os << *fG4Solid;
        }
    }
    void           Print() const override { Print(std::cout); }
    G4GeometryType GetEntityType() const { return fG4Solid->GetEntityType(); }

    vecgeom::VPlacedVolume*
    SpecializedVolume(LogicalVolume const* const    volume,
                      Transformation3D const* const transformation,
                      const TranslationCode,
                      const RotationCode,
                      VPlacedVolume* const /*placement = nullptr*/) const override
    {
        return new GenericPlacedVolume(volume, transformation);
    }

#ifdef VECGEOM_CUDA_INTERFACE
    size_t DeviceSizeOf() const
    {
        return 0;
    }
    DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const
    {
        return {};
    }
    DevicePtr<cuda::VUnplacedVolume>
    CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const) const
    {
        return {};
    }
#endif // VECGEOM_CUDA_INTERFACE

  private:
    const S* fG4Solid;
};
