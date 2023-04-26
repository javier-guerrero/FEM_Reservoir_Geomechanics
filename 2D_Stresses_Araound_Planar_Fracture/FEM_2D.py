import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import gmsh

def stiffness_matrix(D,points,coords):
    assert (len(points) == 3)
    x0, y0, z0 = coords[points[0]]
    x1, y1, z1 = coords[points[1]]
    x2, y2, z2 = coords[points[2]]
    
    Ae = 0.5*abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
    Be = np.array([[y1 - y2,     0.0, y2 - y0,     0.0, y0 - y1,     0.0],
               [    0.0, x2 - x1,     0.0, x0 - x2,     0.0, x1 - x0],
               [x2 - x1, y1 - y2, x0 - x2, y2 - y0, x1 - x0, y0 - y1]])/(2*Ae)
    K = Ae*np.matmul(Be.transpose(), np.matmul(D, Be))
    return K

def assemble_matrix(mesh,D):
    coords,points = mesh
    num_points = len(coords)*2
    K_global = scipy.sparse.csr_matrix((num_points,num_points),dtype=np.float32)
    
    for tri in points:
        K = stiffness_matrix(D,tri,coords)
        ent = np.empty(6,dtype='int')
        ent[0::2] = tri*2
        ent[1::2] = tri*2 + 1
        for i, ind in enumerate(ent):
            for j, jnd in enumerate(ent):
                K_global[ind,jnd] += K[i,j]
    return K_global

def set_bc(K, F, dof, val, dir_):
    if dir_ == 'y':
        dof = 2*dof + 1
    elif dir_ == 'x':
        dof = 2*dof
    K[dof] = 0.0
    K[dof, dof] = 1.0
    F[dof] = val
    
def set_bc_scalar(K, F, dof, val):
    K[dof] = 0.0
    K[dof, dof] = 1.0
    F[dof] = val
    
def boundary_load(mesh=None,F=None,load=None,dof=None,dir_=None):
    x = mesh[0][dof,0]
    y = mesh[0][dof,1]
    z = mesh[0][dof,2]
    num_node = len(dof)
    num_facet = num_node - 1
    dof_ = np.copy(dof)
    if dir_ == 'x':
        A_tot = 1*(np.max(y)-np.min(y))
        idx = np.where((y == np.min(y)) | (y == np.max(y)))[0]
        dof_ *= 2
    elif dir_ == 'y':
        A_tot = 1*(np.max(x)-np.min(x))
        idx = np.where((x == np.min(x)) | (x == np.max(x)))[0]
        dof_ *= 2
        dof_ += 1

    A_facet = A_tot/num_facet
    F[dof_] = load*A_facet
    F[dof[idx]] /= 2
    
def contact_kernel(K,Kn,pairs,nx,ny):
    dof_slave = pairs[:,0]
    dof_master = pairs[:,1]
    
    C = np.array([nx,ny,-nx,-ny])
    kernel = Kn*np.outer(C, C.transpose())

    for j in range(2):
        K[2*dof_slave,2*dof_slave+j] += kernel[0,0+j]
        K[2*dof_slave,2*dof_master+j] += kernel[0,2+j]
        K[2*dof_slave+1,2*dof_slave+j] += kernel[1,0+j]
        K[2*dof_slave+1,2*dof_master+j] += kernel[1,2+j]
        K[2*dof_master,2*dof_slave+j] += kernel[2,0+j]
        K[2*dof_master,2*dof_master+j] += kernel[2,2+j]
        K[2*dof_master+1,2*dof_slave+j] += kernel[3,0+j]
        K[2*dof_master+1,2*dof_master+j] += kernel[3,2+j]

def contact_kernel_scalar(K,hc,pairs):
    dof_slave = pairs[:,0]
    dof_master = pairs[:,1]
    
    K[dof_slave,dof_slave] += hc
    K[dof_slave,dof_master] -= hc
    
    K[dof_master,dof_slave] -= hc
    K[dof_master,dof_master] += hc
        
def stiffness_matrix_scalar(D,points,coords):
    assert (len(points) == 3)
    x0, y0, z0 = coords[points[0]]
    x1, y1, z1 = coords[points[1]]
    x2, y2, z2 = coords[points[2]]
    
    Ae = 0.5*abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
    B = np.array([[y1 - y2, y2 - y0, y0 - y1],
               [x2 - x1, x0 - x2, x1 - x0]])/(2*Ae)
    K = Ae*np.matmul(B.transpose(), np.matmul(D, B))
    return K

def capacitance_matrix_scalar(D,points,coords):
    assert (len(points) == 3)
    x0, y0, z0 = coords[points[0]]
    x1, y1, z1 = coords[points[1]]
    x2, y2, z2 = coords[points[2]]
    
    Ae = 0.5*abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
    phi = np.array([[2,1,1],[1,2,1],[1,1,2]])
    C = Ae*D/12*phi
    return C

def assemble_capacitance_matrix(mesh,D):
    coords,points = mesh
    num_points = len(coords)
    C_global = scipy.sparse.csr_matrix((num_points,num_points),dtype=np.float32)
    
    for tri in points:
        C = capacitance_matrix_scalar(D,tri,coords)
        for i, ind in enumerate(tri):
            for j, jnd in enumerate(tri):
                C_global[ind,jnd] += C[i,j]
    return C_global

def assemble_matrix_scalar(mesh,D):
    coords,points = mesh
    num_points = len(coords)
    K_global = scipy.sparse.csr_matrix((num_points,num_points),dtype=np.float32)
    
    for tri in points:
        K = stiffness_matrix_scalar(D,tri,coords)
        for i, ind in enumerate(tri):
            for j, jnd in enumerate(tri):
                K_global[ind,jnd] += K[i,j]
    return K_global

def symm_gradient(points,coords):
    assert (len(points) == 3)
    x0, y0, z0 = coords[points[0]]
    x1, y1, z1 = coords[points[1]]
    x2, y2, z2 = coords[points[2]]
    
    Ae = 0.5*abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
    Be = np.array([[y1 - y2,     0.0, y2 - y0,     0.0, y0 - y1,     0.0],
               [    0.0, x2 - x1,     0.0, x0 - x2,     0.0, x1 - x0],
               [x2 - x1, y1 - y2, x0 - x2, y2 - y0, x1 - x0, y0 - y1]])/(2*Ae)
    return Be,Ae

def assemble_stress_strain(mesh,u,D):
    coords,points = mesh
    strain = np.zeros((len(points),3))
    stress = np.zeros((len(points),3))
    
    for (i,tri) in enumerate(points):
        Be,Ae = symm_gradient(tri,coords)
        uu = np.array([u[int(2*tri[0])],u[int(2*tri[0]+1)],
                       u[int(2*tri[1])],u[int(2*tri[1]+1)],
                       u[int(2*tri[2])],u[int(2*tri[2]+1)]])
        strain[i,:] = np.matmul(Be,uu)
        stress[i,:] = np.matmul(D,strain[i,:])
    return strain,stress
        
def plot(mesh, title, data=None,zoom=False):
    coord,node = mesh
    x,y,z = coord[:,0],coord[:,1],coord[:,2]
    
    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    if zoom:
        ax.set_xlim([-15,15])
        ax.set_ylim([-15,15])
    else:
        ax.set_xlim([x.min()-1,x.max()+1])
        ax.set_ylim([y.min()-1,y.max()+1])
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    if data is not None:
        if len(data) == len(coord):
            mag = data
            tric = ax.tricontourf(x,y,node,mag,len(mag),cmap='Spectral')
            cbar = plt.colorbar(tric, shrink=0.8)
            
        elif len(data) == int(2*len(coord)):
            ux = data[0::2]
            uy = data[1::2]
            mag = np.sqrt(ux**2+uy**2)
            tric = ax.tricontourf(x,y,node,mag,len(mag),cmap='Spectral')
            cbar = plt.colorbar(tric, shrink=0.8)
            
        elif len(data) == len(node):
            trip = ax.tripcolor(x,y,node,facecolors=data,edgecolors='None',cmap='Spectral')
            cbar = plt.colorbar(trip, shrink=0.8)
            
    ax.triplot(x,y,node,color='k',linewidth=0.5)
    plt.show()
    return None

def consolidation_mesh(name=None,width=None,height=None,ms=None):
    gmsh.initialize()
    gmsh.model.add(name)
    
    # boundary nodes
    ll = gmsh.model.occ.addPoint(0,0,0,ms,-1)
    lr = gmsh.model.occ.addPoint(width,0,0,ms,-1)
    ur = gmsh.model.occ.addPoint(width,height,0,ms,-1)
    ul = gmsh.model.occ.addPoint(0,height,0,ms,-1)
    
    # boundary lines
    bt = gmsh.model.occ.addLine(ll,lr,-1)
    rt = gmsh.model.occ.addLine(lr,ur,-1)
    tp = gmsh.model.occ.addLine(ur,ul,-1)
    lt = gmsh.model.occ.addLine(ul,ll,-1)
    
    # boundary curve loop
    bd = gmsh.model.occ.addCurveLoop([bt,rt,tp,lt],-1)
    
    # boundary surface
    s = gmsh.model.occ.addPlaneSurface([bd])
    gmsh.model.occ.synchronize()
    
    gmsh.model.mesh.generate(2)
    
    cell_types, cell_tags, cell_node_tags = gmsh.model.mesh.getElements(dim=2)
    cell_tags = cell_tags[0]
    cell_node_tags = cell_node_tags[0].reshape((len(cell_tags),3))
    cell_node_tags -= 1
    
    node_tags, coord, param_coords = gmsh.model.mesh.getNodes()
    coord = coord.reshape((int(len(coord)/3),3))
    sorted_ind = np.lexsort((coord[:,2], coord[:,1],coord[:,0]))
    
    gmsh.finalize()
    mesh = (coord,cell_node_tags)
    return mesh

def fracture_mesh(name=None,width=None,height=None,ms=None,mf=None):
    gmsh.initialize()
    gmsh.model.add(name)

    # boundary nodes
    ll = gmsh.model.occ.addPoint(0,0,0,ms,1)
    lr = gmsh.model.occ.addPoint(width,0,0,ms,2)
    ur = gmsh.model.occ.addPoint(width,height,0,ms,3)
    ul = gmsh.model.occ.addPoint(0,height,0,ms,4)

    # boundary lines
    bt = gmsh.model.occ.addLine(ll,lr,1)
    rt = gmsh.model.occ.addLine(lr,ur,2)
    tp = gmsh.model.occ.addLine(ur,ul,3)
    lt = gmsh.model.occ.addLine(ul,ll,4)

    # boundary curve loop
    bd = gmsh.model.occ.addCurveLoop([bt,rt,tp,lt],1)

    # boundary surface
    s = gmsh.model.occ.addPlaneSurface([bd])

    # fracture geometry
    fl = gmsh.model.occ.addPoint(width*0.5,height*0.40,0,mf,5) # 0.35
    fr = gmsh.model.occ.addPoint(width*0.5,height*0.60,0,mf,6) # 0.65

    fe = gmsh.model.occ.addLine(fl,fr)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.embed(1,[fe],2,s)
    gmsh.model.mesh.embed(0,[fl,fr],2,s)

    gmsh.model.mesh.generate(2)
    gdim = 2
    gmsh.model.addPhysicalGroup(gdim-1, [bt],tag=1)
    gmsh.model.setPhysicalName(gdim-1, 1, 'Base')
    gmsh.model.addPhysicalGroup(gdim-1, [rt],tag=2)
    gmsh.model.setPhysicalName(gdim-1, 2, 'Right')
    gmsh.model.addPhysicalGroup(gdim-1, [tp],tag=3)
    gmsh.model.setPhysicalName(gdim-1, 3, 'Top')
    gmsh.model.addPhysicalGroup(gdim-1, [lt],tag=4)
    gmsh.model.setPhysicalName(gdim-1, 4, 'Left')
    gmsh.model.addPhysicalGroup(gdim-1, [fe],tag=5)
    gmsh.model.addPhysicalGroup(gdim, [s],tag=10)
    gmsh.model.setPhysicalName(gdim, 10, 'Omega')

    gmsh.model.addPhysicalGroup(gdim-2, [ur],tag=20)
    gmsh.model.setPhysicalName(gdim-2, 20, 'Point')
    
    gmsh.plugin.setNumber('Crack','Dimension',1)
    gmsh.plugin.setNumber('Crack','PhysicalGroup',5)
    gmsh.plugin.setNumber('Crack','NewPhysicalGroup',6)
    gmsh.plugin.run("Crack")
    gmsh.model.setPhysicalName(gdim-1, 5, 'Fracture_minus')
    gmsh.model.setPhysicalName(gdim-1, 6, 'Fracture_plus')
    
    gmsh.model.mesh.renumberNodes()
    gmsh.model.occ.synchronize()
    
    cell_types, cell_tags, cell_node_tags = gmsh.model.mesh.getElements(dim=2)
    cell_tags = cell_tags[0]
    cell_node_tags = cell_node_tags[0].reshape((len(cell_tags),3))
    cell_node_tags -= 1
    
    node_tags, coord, param_coords = gmsh.model.mesh.getNodes()
    coord = coord.reshape((int(len(coord)/3),3))
    
    node_slave,coord_slave = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)
    node_master,coord_master = gmsh.model.mesh.getNodesForPhysicalGroup(1,6)
    
    coord_slave = coord_slave.reshape((int(len(coord_slave)/3),3))
    coord_master = coord_master.reshape((int(len(coord_master)/3),3))

    gmsh.finalize()
    
    k = 0
    h = np.zeros(len(cell_node_tags))
    for tri in cell_node_tags:
        x0, y0, z0 = coord[tri[0]]
        x1, y1, z1 = coord[tri[1]]
        x2, y2, z2 = coord[tri[2]]
        a = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
        b = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        c = np.sqrt((x0-x2)**2+(y0-y2)**2+(z0-z2)**2)
        s = 0.5*(a+b+c)
        h[k] = 2*a*b*c/4/np.sqrt(s*(s-a)*(s-b)*(s-c))
        k += 1

    mesh = (coord,cell_node_tags)
    contact_node,contact_coord = (node_slave,node_master), (coord_slave,coord_master)
    pairs = np.zeros([len(contact_node[0]),2],dtype=int)
    
    for i in range(len(contact_node[0])):
        for j in range(len(contact_node[1])):
            if np.all(np.isclose(coord_slave[i,:],coord_master[j,:])):
                pairs[i,0] = contact_node[0][i] - 1
                pairs[i,1] = contact_node[1][j] - 1

    idx = np.where(pairs[:,0] == pairs[:,1])[0]
    pairs = np.delete(pairs,idx,axis=0)
    
    return mesh, pairs, h
