//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootInterfaceLinkDef.h
//! \brief Define the classes added to the ROOT dictionary
//---------------------------------------------------------------------------//
#ifdef __ROOTCLING__

// clang-format off
// Import data
#pragma link C++ class celeritas::ImportAtomicRelaxation+;
#pragma link C++ class celeritas::ImportAtomicSubshell+;
#pragma link C++ class celeritas::ImportAtomicTransition+;
#pragma link C++ class celeritas::ImportData+;
#pragma link C++ class celeritas::ImportData::ImportAtomicRelaxationMap+;
#pragma link C++ class celeritas::ImportData::ImportLivermorePEMap+;
#pragma link C++ class celeritas::ImportData::ImportSBMap+;
#pragma link C++ class celeritas::ImportElement+;
#pragma link C++ class celeritas::ImportEmParameters+;
#pragma link C++ class celeritas::ImportGeoMaterial+;
#pragma link C++ class celeritas::ImportIsotope+;
#pragma link C++ class celeritas::ImportLivermorePE+;
#pragma link C++ class celeritas::ImportLivermoreSubshell+;
#pragma link C++ class celeritas::ImportLoopingThreshold+;
#pragma link C++ class celeritas::ImportMatElemComponent+;
#pragma link C++ class celeritas::ImportMaterialScintSpectrum+;
#pragma link C++ class celeritas::ImportModel+;
#pragma link C++ class celeritas::ImportModelMaterial+;
#pragma link C++ class celeritas::ImportModelMaterial::EnergyBound+;
#pragma link C++ class celeritas::ImportMscModel+;
#pragma link C++ class celeritas::ImportMuPairProductionTable+;
#pragma link C++ class celeritas::ImportOpticalMaterial+;
#pragma link C++ class celeritas::ImportOpticalModel+;
#pragma link C++ class celeritas::ImportOpticalParameters+;
#pragma link C++ class celeritas::ImportOpticalProperty+;
#pragma link C++ class celeritas::ImportOpticalRayleigh+;
#pragma link C++ class celeritas::ImportParticle+;
#pragma link C++ class celeritas::ImportParticleScintSpectrum+;
#pragma link C++ class celeritas::ImportPhysicsTable+;
#pragma link C++ class celeritas::ImportPhysMaterial+;
#pragma link C++ class celeritas::ImportProcess+;
#pragma link C++ class celeritas::ImportProductionCut+;
#pragma link C++ class celeritas::ImportRegion+;
#pragma link C++ class celeritas::ImportScintComponent+;
#pragma link C++ class celeritas::ImportScintData+;
#pragma link C++ class celeritas::ImportTransParameters+;
#pragma link C++ class celeritas::ImportVolume+;
#pragma link C++ class celeritas::ImportWavelengthShift+;

// Input data
#pragma link C++ class celeritas::inp::GaussianRoughness+;
#pragma link C++ class celeritas::inp::FresnelReflection+;
#pragma link C++ class celeritas::inp::Grid+;
#pragma link C++ class celeritas::inp::GridReflection+;
#pragma link C++ class celeritas::inp::InteractionModels+;
#pragma link C++ class celeritas::inp::Interpolation+;
#pragma link C++ class celeritas::inp::NoRoughness+;
#pragma link C++ class celeritas::inp::OpticalPhysics+;
#pragma link C++ class celeritas::inp::ReflectionForm+;
#pragma link C++ class celeritas::inp::ReflectivityModels+;
#pragma link C++ class celeritas::inp::RoughnessModels+;
#pragma link C++ class celeritas::inp::SmearRoughness+;
#pragma link C++ class celeritas::inp::SurfacePhysics+;
#pragma link C++ class celeritas::inp::TwodGrid+;
#pragma link C++ class celeritas::inp::UniformGrid+;
#pragma link C++ class celeritas::inp::UniformGrid::GridBound+;

// Event data used by Geant4/Celeritas offloading applications
#pragma link C++ class celeritas::EventHitData+;
#pragma link C++ class celeritas::EventData+;

// List of OpaqueIds
#pragma link C++ class celeritas::SurfaceId+;
// clang-format on

#endif
