//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HitRootIO.cc
//---------------------------------------------------------------------------//
#include "HitRootIO.hh"

#include <cstdio>
#include <regex>
#include <G4Event.hh>

#if G4VERSION_NUMBER < 1070
#    include <celeritas/ext/GeantSetup.hh>
#endif

#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Version.hh>
#include <TBranch.h>
#include <TFile.h>
#include <TObject.h>
#include <TROOT.h>
#include <TTree.h>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveHit.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Create a ROOT output file for each worker except the master thread if MT.
 */
HitRootIO::HitRootIO()
{
    ROOT::EnableThreadSafety();

    file_name_ = std::regex_replace(
        GlobalSetup::Instance()->GetSetupOptions()->output_file,
        std::regex("\\.json$"),
        ".root");

    if (G4Threading::IsWorkerThread())
    {
        file_name_ += std::to_string(G4Threading::G4GetThreadId());
    }

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        file_.reset(TFile::Open(file_name_.c_str(), "recreate"));
        CELER_VALIDATE(file_->IsOpen(), << "failed to open " << file_name_);
        tree_.reset(new TTree(
            "Events", "Hit collections", this->SplitLevel(), file_.get()));
    }
}

//---------------------------------------------------------------------------//
//! Default destructor
HitRootIO::~HitRootIO() = default;

//---------------------------------------------------------------------------//
/*!
 * Return the static thread local singleton instance>
 */
HitRootIO* HitRootIO::Instance()
{
    static G4ThreadLocal HitRootIO instance;
    return &instance;
}

//---------------------------------------------------------------------------//
/*!
 * Write sensitive hits to output in the form of HitRootEvent.
 */
void HitRootIO::WriteHits(G4Event const* event)
{
    G4HCofThisEvent* hce = event->GetHCofThisEvent();
    if (hce == nullptr)
    {
        return;
    }

    // Write the collection of sensitive hits into HitRootEvent
    HitRootEvent hit_event;
    hit_event.event_id = event->GetEventID();
    for (int i = 0; i < hce->GetNumberOfCollections(); i++)
    {
        G4VHitsCollection* hc = hce->GetHC(i);
        std::string hcname = hc->GetName();
        std::vector<G4VHit*> hits;
        int number_of_hits = hc->GetSize();
        for (int j = 0; j < number_of_hits; ++j)
        {
            G4VHit* hit = hc->GetHit(j);
            SensitiveHit* sd_hit = dynamic_cast<SensitiveHit*>(hit);
            hits.push_back(sd_hit);
        }
        hit_event.hcmap.insert(std::make_pair(hcname, std::move(hits)));
    }

    // Write a HitRootEvent into output ROOT file
    this->WriteObject(&hit_event);
}

//---------------------------------------------------------------------------//
/*!
 * Fill a HitRootEvent object.
 */
void HitRootIO::WriteObject(HitRootEvent* hit_event)
{
    if (!event_branch_)
    {
        event_branch_
            = tree_->Branch("event.",
                            &hit_event,
                            GlobalSetup::Instance()->GetRootBufferSize(),
                            this->SplitLevel());
    }
    else
    {
        event_branch_->SetAddress(&hit_event);
    }

    tree_->Fill();
    event_branch_->ResetAddress();
}

//---------------------------------------------------------------------------//
/*!
 * Write, and Close or Merge output.
 */
void HitRootIO::Close()
{
    CELER_LOG_LOCAL(status) << "Closing ROOT file";
    CELER_EXPECT((file_ && file_->IsOpen())
                 || (G4Threading::IsMultithreadedApplication()
                     && G4Threading::IsMasterThread()));

    if (!G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG(info) << "Writing hit ROOT output to " << file_name_;
        CELER_ASSERT(tree_);
        file_->Write("", TObject::kOverwrite);
    }
    else
    {
        if (G4Threading::IsMasterThread())
        {
            // Merge output file on the master thread
            this->Merge();
        }
        else
        {
            file_->Write("", TObject::kOverwrite);
        }
    }
    event_branch_ = nullptr;
    tree_.reset();
    file_.reset();
}

//---------------------------------------------------------------------------//
/*!
 * Merging output root files from multiple threads using TTree::MergeTrees.
 *
 * TODO: use TBufferMerger and follow the example described in the ROOT
 * tutorials/multicore/mt103_fillNtupleFromMultipleThreads.C which stores
 * TBuffer data in memory and writes 32MB compressed output concurrently.
 */
void HitRootIO::Merge()
{
#if G4VERSION_NUMBER >= 1070
    auto const nthreads = G4RunManager::GetRunManager()->GetNumberOfThreads();
#else
    auto const nthreads
        = celeritas::GetNumberOfThreads(*G4RunManager::GetRunManager());
#endif
    std::vector<TFile*> files;
    std::vector<TTree*> trees;
    std::unique_ptr<TList> list(new TList);

    CELER_LOG_LOCAL(info) << "Merging hit root files from " << nthreads
                          << " threads into \"" << file_name_ << "\"";

    for (int i = 0; i < nthreads; ++i)
    {
        std::string file_name = file_name_ + std::to_string(i);
        files.push_back(TFile::Open(file_name.c_str()));
        trees.push_back((TTree*)(files[i]->Get("Events")));
        list->Add(trees[i]);

        if (i == nthreads - 1)
        {
            TFile* file = TFile::Open(file_name_.c_str(), "recreate");
            CELER_VALIDATE(file->IsOpen(), << "failed to open " << file_name_);

            TTree* tree = TTree::MergeTrees(list.get());
            tree->SetName("Events");
            //  Write both the TFile and TTree meta-data
            file->Write();
            file->Close();
        }
        // Delete the merged file
        std::remove(file_name.c_str());
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
