//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file GenericSolid.hh
 * \brief Class for a generic solid related to G4 to VecGeom conversion
 *
 * Original class VecGeomG4Solid, from G4VecGeomNav package by John Apostolakis
 *   et.al., and adapted to Celeritas.
 *
 * Original source:
 *   https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/ce4a0bb2f777a6728e59dee96c764fd61fa8b785/include/VecGeomG4Solid.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <G4VSolid.hh>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/UnplacedVolume.h>

#include "GenericPlacedVolume.hh"
using namespace vecgeom;

namespace celeritas
{
/*!
 * A generic VecGeom solid converted from a Geant4 geometry.
 *
 * Some complex geometry functionalities in Geant4 don't have equivalents yet
 * in VecGeom. This class can be used either as a placeholder, or as a wrapper
 * when it makes sense to use some functionality from the original G4 shape.
 */
template<typename S>
class GenericSolid : public VUnplacedVolume
{
  private:
    //!@{
    //! \name Type aliases
    using VUnplacedVolume::DistanceToIn;
    using VUnplacedVolume::DistanceToOut;
    using VUnplacedVolume::SafetyToIn;
    using VUnplacedVolume::SafetyToOut;
    //!@}

  public:
    explicit GenericSolid(S const* g4solid) : g4_solid_(g4solid) {}

    static G4ThreeVector ToG4V(Vector3D<double> const& p)
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

    // ---------------- Contains --------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    bool Contains(Vector3D<Precision> const& p) const override
    {
        auto loc = this->ConvertEnum(g4_solid_->Inside(ToG4V(p)));
        return !(loc == EnumInside::kOutside);
    }

    VECCORE_ATT_HOST_DEVICE
    EnumInside Inside(Vector3D<Precision> const& p) const override
    {
        return this->ConvertEnum(g4_solid_->Inside(ToG4V(p)));
    }

    // ---------------- DistanceToOut functions ------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision DistanceToOut(Vector3D<Precision> const& p,
                            Vector3D<Precision> const& d,
                            Precision /*step_max = kInfLength*/) const override
    {
        return g4_solid_->DistanceToOut(ToG4V(p), ToG4V(d));
    }

    // USolid/GEANT4-like interface for DistanceToOut (returns also
    // exiting normal)
    VECCORE_ATT_HOST_DEVICE
    Precision DistanceToOut(Vector3D<Precision> const& p,
                            Vector3D<Precision> const& d,
                            Vector3D<Precision>& normal,
                            bool& convex,
                            Precision /*step_max = kInfLength*/) const override
    {
        bool calculateNorm = true;
        G4ThreeVector normalG4;
        auto dist = g4_solid_->DistanceToOut(
            ToG4V(p), ToG4V(d), calculateNorm, &convex, &normalG4);
        normal = Vector3D<Precision>(normalG4[0], normalG4[1], normalG4[2]);
        return dist;
    }

    // ---------------- SafetyToOut functions -------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision SafetyToOut(Vector3D<Precision> const& p) const override
    {
        return g4_solid_->DistanceToOut(ToG4V(p));
    }

    // ---------------- DistanceToIn functions ------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision
    DistanceToIn(Vector3D<Precision> const& p,
                 Vector3D<Precision> const& d,
                 const Precision /*step_max = kInfLength*/) const override
    {
        return g4_solid_->DistanceToIn(ToG4V(p), ToG4V(d));
    }

    // ---------------- SafetyToIn functions ---------------------------------
    VECCORE_ATT_HOST_DEVICE
    Precision SafetyToIn(Vector3D<Precision> const& p) const override
    {
        return g4_solid_->DistanceToIn(ToG4V(p));
    }

    // ---------------- Normal ---------------------------------------------

    VECCORE_ATT_HOST_DEVICE
    bool Normal(Vector3D<Precision> const& p,
                Vector3D<Precision>& normal) const override
    {
        auto n = g4_solid_->SurfaceNormal(ToG4V(p));
        normal = Vector3D<double>(n[0], n[1], n[2]);
        return true;
    }

    // ----------------- Extent --------------------------------------------
    VECCORE_ATT_HOST_DEVICE
    void
    Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override
    {
        auto ext = g4_solid_->GetExtent();
        aMin.Set(ext.GetXmin(), ext.GetYmin(), ext.GetZmin());
        aMax.Set(ext.GetXmax(), ext.GetYmax(), ext.GetZmax());
    }

    double Capacity() const override
    {
        return const_cast<S*>(g4_solid_)->S::GetCubicVolume();
    }

    double SurfaceArea() const override
    {
        return const_cast<S*>(g4_solid_)->S::GetSurfaceArea();
    }

    int MemorySize() const override { return sizeof(this); }
    void Print(std::ostream& os) const override
    {
        if (g4_solid_)
        {
            os << *g4_solid_;
        }
    }

    void Print() const override { Print(std::cout); }
    G4GeometryType GetEntityType() const { return g4_solid_->GetEntityType(); }

    VPlacedVolume*
    SpecializedVolume(LogicalVolume const* const volume,
                      Transformation3D const* const transformation,
                      const TranslationCode,
                      const RotationCode,
                      VPlacedVolume* const /*placement = nullptr*/) const override
    {
        return new GenericPlacedVolume(volume, transformation);
    }

#ifdef VECGEOM_CUDA_INTERFACE
    //! @{ /name Required interface
    //
    // These implementations are required when CUDA is enabled.
    // A trivialimplementation is okay, since won't be called from GPU.
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
    //! @}
#endif

  private:
    //!@{ \name Deleted constructors and assignment operator.
    ~GenericSolid() = default;
    GenericSolid(GenericSolid const&) = delete;
    GenericSolid(GenericSolid const&&) = delete;
    GenericSolid& operator=(GenericSolid const&) = delete;
    //!@}

  private:
    S const* g4_solid_;
};

}  // namespace celeritas
