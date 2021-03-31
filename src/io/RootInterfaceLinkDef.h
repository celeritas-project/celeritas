//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootInterfaceLinkDef.h
//! Define the classes added to the ROOT dictionary.
//---------------------------------------------------------------------------//
#ifdef __CINT__

// clang-format off
#pragma link C++ class celeritas::ImportParticle+;
#pragma link C++ class celeritas::ImportProcess+;
#pragma link C++ class celeritas::ImportPhysicsTable+;
#pragma link C++ class celeritas::ImportPhysicsVector+;
#pragma link C++ class celeritas::GdmlGeometryMap+;
#pragma link C++ class celeritas::ImportMaterial+;
#pragma link C++ class celeritas::ImportMaterial::ProductionCut+;
#pragma link C++ class celeritas::ImportElement+;
#pragma link C++ class celeritas::ImportVolume+;
#pragma link C++ class celeritas::RootLoader+;
// clang-format on

#endif
