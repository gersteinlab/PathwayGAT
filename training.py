from utils import *
from model import PathwayGAT
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import pickle

def training_g(pathway_file, gene_file, meta_file, class_name, output_prefix, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse gene expression value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    subset_expr, opgene = parse_gene_expression(gene_file, wpgene, None)

    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(subset_expr.shape[0]), sample_num)
    else:
        sample_list = range(subset_expr.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    subset_expr = subset_expr.iloc[sample_list, ]

    # Construct dataset and target
    dataset1g = construct_pathnodes1g(wpdict, subset_expr)
    torch.save(dataset1g, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train1g = create_geometric_dataset(dataset1g, wp_edge, label_df['label'])
    torch.save(dt4train1g, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train1g, [int(0.8 * len(dt4train1g)), int(0.2 * len(dt4train1g))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset1g.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")


def training_m(pathway_file, microbe_file, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse microbe abundance value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    microbe_gene_corr = read_microbe_gene_corr(microbe_gene_file, 2)
    microbe_abundance = pd.read_csv(microbe_file, header=0, delimiter='\t', index_col=0)
    
    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(microbe_abundance.shape[0]), sample_num)
    else:
        sample_list = range(microbe_abundance.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    microbe_abundance = microbe_abundance.iloc[sample_list, ]

    # Construct dataset and target
    microbe_dict = generate_microbe_features(microbe_abundance, microbe_gene_corr, microbe_corr_threshold, wpdict)
    dataset1m = construct_pathnodes1m(wpdict, microbe_dict)
    torch.save(dataset1m, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train1m = create_geometric_dataset(dataset1m, wp_edge, label_df['label'])
    torch.save(dt4train1m, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train1m, [int(0.8 * len(dt4train1m)), int(0.2 * len(dt4train1m))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset1m.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
        
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")

def training_mg(pathway_file, microbe_file, gene_file, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse gene/microbe abundance value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    microbe_gene_corr = read_microbe_gene_corr(microbe_gene_file, 2)
    microbe_abundance = pd.read_csv(microbe_file, header=0, delimiter='\t', index_col=0)

    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(microbe_abundance.shape[0]), sample_num)
    else:
        sample_list = range(microbe_abundance.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    microbe_abundance = microbe_abundance.iloc[sample_list, ]
    
    microbe_dict = generate_microbe_features(microbe_abundance, microbe_gene_corr, microbe_corr_threshold, wpdict)
    subset_expr, opgene = parse_gene_expression(gene_file, wpgene, sample_list)
    
    # Construct dataset and target
    dataset2 = construct_pathnodes2(wpdict, subset_expr, microbe_dict)
    torch.save(dataset2, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train2 = create_geometric_dataset(dataset2, wp_edge, label_df['label'])
    torch.save(dt4train2, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train2, [int(0.8 * len(dt4train2)), int(0.2 * len(dt4train2))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset2.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
        
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")

def training_msnp(pathway_file, microbe_file, SNP_file, SNP_coding, SNP_noncoding, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse SNP/microbe information value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    microbe_gene_corr = read_microbe_gene_corr(microbe_gene_file, 2)
    microbe_abundance = pd.read_csv(microbe_file, header=0, delimiter='\t', index_col=0)

    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(microbe_abundance.shape[0]), sample_num)
    else:
        sample_list = range(microbe_abundance.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    microbe_abundance = microbe_abundance.iloc[sample_list, ]
    
    microbe_dict = generate_microbe_features(microbe_abundance, microbe_gene_corr, microbe_corr_threshold, wpdict)
    snp_ncd_cd = snp_info(SNP_coding, SNP_noncoding, SNP_file, wpdict, sample_list)
    
    # Construct dataset and target
    dataset3 = construct_pathnodes3msnp(wpdict, microbe_dict, snp_ncd_cd)
    torch.save(dataset3, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train3 = create_geometric_dataset(dataset3, wp_edge, label_df['label'])
    torch.save(dt4train3, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train3, [int(0.8 * len(dt4train3)), int(0.2 * len(dt4train3))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset3.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
        
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")

def training_gsnp(pathway_file, gene_file, SNP_file, SNP_coding, SNP_noncoding, meta_file, class_name, output_prefix, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse SNP/gene information value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    subset_expr, opgene = parse_gene_expression(gene_file, wpgene, None)

    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(subset_expr.shape[0]), sample_num)
    else:
        sample_list = range(subset_expr.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    subset_expr = subset_expr.iloc[sample_list, ]
    
    snp_ncd_cd = snp_info(SNP_coding, SNP_noncoding, SNP_file, wpdict, sample_list)

    # Construct dataset and target
    dataset3 = construct_pathnodes3gsnp(wpdict, subset_expr, snp_ncd_cd)
    torch.save(dataset3, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train3 = create_geometric_dataset(dataset3, wp_edge, label_df['label'])
    torch.save(dt4train3, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train3, [int(0.8 * len(dt4train3)), int(0.2 * len(dt4train3))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset3.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
        
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")


def training_mgsnp(pathway_file, microbe_file, gene_file, SNP_file, SNP_coding, SNP_noncoding, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed):
    # Set seed for all packages
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Construct the pathway network and parse SNP/gene/microbe information value to the network
    wpgene, wpdict = parse_pathway_file(pathway_file)
    wpadj = calculate_adjacency(wpdict)
    wp_edge = adjacency_to_edge_index(wpadj)
    
    microbe_gene_corr = read_microbe_gene_corr(microbe_gene_file, 2)
    microbe_abundance = pd.read_csv(microbe_file, header=0, delimiter='\t', index_col=0)

    # Downsample if necessary
    if sample_num > 0:
        sample_list = random.sample(range(microbe_abundance.shape[0]), sample_num)
    else:
        sample_list = range(microbe_abundance.shape[0])
    with open(f'{output_prefix}_sample_index.pkl', 'wb') as f:
        pickle.dump(sample_list, f)
    microbe_abundance = microbe_abundance.iloc[sample_list, ]
    
    microbe_dict = generate_microbe_features(microbe_abundance, microbe_gene_corr, microbe_corr_threshold, wpdict)
    
    subset_expr, opgene = parse_gene_expression(gene_file, wpgene, sample_list)
    snp_ncd_cd = snp_info(SNP_coding, SNP_noncoding, SNP_file, wpdict, sample_list)

    # Construct dataset and target
    dataset4 = construct_pathnodes4(wpdict, subset_expr, microbe_dict, snp_ncd_cd)
    torch.save(dataset4, f'{output_prefix}_nodes.pt')
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    dt4train4 = create_geometric_dataset(dataset4, wp_edge, label_df['label'])
    torch.save(dt4train4, f'{output_prefix}_training.pt')
    
    train_dataset, test_dataset = random_split(dt4train4, [int(0.8 * len(dt4train4)), int(0.2 * len(dt4train4))])
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training, validation, and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PathwayGAT(num_features=dataset4.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label']))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    outfn = ""
    for epoch in range(epochs):
        # Training
        model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        train_loss = loss_all / (len(train_loader)+1)
        
        # Evaluation
        model.eval()
        loss_all = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                output = model(data.x, data.edge_index, data.batch, device)
                loss_all += criterion(output, data.y).item()
                _, preds = output.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            val_loss = loss_all / (len(val_loader)+1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc = accuracy_score(all_labels, all_preds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Best so far! Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
            outfn = f'{output_prefix}.{epoch}.best_model.pth'
            torch.save(model.state_dict(), outfn)
            outfn = f'{output_prefix}.best_model.pth'
            torch.save(model.state_dict(), outfn)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}')
        
    outfn = f'{output_prefix}.final_model.pth'
    torch.save(model.state_dict(), outfn)
    # Test
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.to(torch.float32)
            output = model(data.x, data.edge_index, data.batch, device)       
            test_loss += criterion(output, data.y).item()
            _, preds = output.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {test_loss}, Acc: {acc:.4f}")
